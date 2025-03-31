from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math

# =============================================================================
# ========================= Dataset and Tasks =================================
# =============================================================================

# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(bsz=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

Datasets = {
    "mnist": create_mnist_dataset,
    "mnist-classification": create_mnist_classification_dataset,
}

# =============================================================================
# ========================= State Space Model (SSM)  ==========================
# =============================================================================

class SSMKernel(nn.Module):
    """
    PyTorch implementation of the SSM Kernel
    """
    def __init__(self, N, l_max, dt_min=0.001, dt_max=0.1):
        super().__init__()
        # SSM parameters
        self.N = N
        self.l_max = l_max
        
        # Initialize parameters with PyTorch
        self.A = nn.Parameter(torch.randn(N, N) / math.sqrt(N))
        self.B = nn.Parameter(torch.randn(N, 1) / math.sqrt(N))
        self.C = nn.Parameter(torch.randn(1, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(1))
        
        # Step parameter
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Register buffer for RNN cache
        self.register_buffer('x_k_1', torch.zeros(N))
        
        # Precompute kernel for CNN mode
        self.kernel = None
        
    def discretize(self):
        dt = torch.exp(self.log_dt)
        I = torch.eye(self.N, device=self.A.device)
        
        # Compute the discrete-time matrices
        BL = torch.inverse(I - (dt / 2.0) * self.A)
        Ab = BL @ (I + (dt / 2.0) * self.A)
        Bb = (BL * dt) @ self.B
        
        return Ab, Bb, self.C
    
    def compute_kernel(self):
        """Compute the convolution kernel for CNN mode"""
        Ab, Bb, C = self.discretize()
        
        # Compute powers of Ab for each step in the sequence
        kernel = []
        X = Bb
        for l in range(self.l_max):
            kernel.append((C @ X).reshape(-1))
            X = Ab @ X
            
        return torch.stack(kernel)
    
    def forward(self, u, decode=False):
        if not decode:
            # CNN Mode - use convolution
            if self.kernel is None or self.training:
                self.kernel = self.compute_kernel()
                
            # Implement causal convolution using PyTorch's conv1d
            # Reshape kernel to [out_channels, in_channels, kernel_size]
            kernel = self.kernel.reshape(1, 1, -1)
            
            # Pad the input for causal convolution
            u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (self.l_max-1, 0))
            
            # Apply convolution and remove extra dimensions
            y = F.conv1d(u_pad, kernel).squeeze(0).squeeze(0)
            
            return y + self.D * u
        else:
            # RNN Mode - for autoregressive generation
            Ab, Bb, C = self.discretize()
            x_k = self.x_k_1
            outputs = []
            
            for u_t in u:
                x_k = Ab @ x_k + Bb * u_t
                y_k = (C @ x_k).reshape(-1)
                outputs.append(y_k)
                
            # Update the cached state
            if self.training:
                self.x_k_1 = x_k.detach()
                
            return torch.stack(outputs) + self.D * u


class SSMLayer(nn.Module):
    """
    Parallel SSM layer that processes multiple channels
    """
    def __init__(self, d_model, N, l_max, decode=False):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.l_max = l_max
        self.decode = decode
        
        # Create parallel SSM kernels for each channel
        self.ssm_kernels = nn.ModuleList([
            SSMKernel(N, l_max) for _ in range(d_model)
        ])
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        seq_len, batch_size, d_model = x.shape
        
        # Process each batch separately
        batch_outputs = []
        for b in range(batch_size):
            # Process each channel with its own SSM kernel
            channel_outputs = []
            for i, kernel in enumerate(self.ssm_kernels):
                u = x[:, b, i]  # [seq_len]
                y = kernel(u, decode=self.decode)
                channel_outputs.append(y)
                
            # Stack channel outputs
            batch_output = torch.stack(channel_outputs, dim=1)  # [seq_len, d_model]
            batch_outputs.append(batch_output)
            
        # Stack batch outputs
        return torch.stack(batch_outputs, dim=1)  # [seq_len, batch_size, d_model]


class SequenceBlock(nn.Module):
    """
    Sequence processing block with normalization, SSM layer, and MLP
    """
    def __init__(self, d_model, N, l_max, dropout=0.0, prenorm=True, glu=True, decode=False):
        super().__init__()
        self.prenorm = prenorm
        self.glu = glu
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # SSM layer
        self.seq = SSMLayer(d_model, N, l_max, decode=decode)
        
        # Output projections
        self.out = nn.Linear(d_model, d_model)
        if glu:
            self.out2 = nn.Linear(d_model, d_model)
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        skip = x
        
        if self.prenorm:
            x = self.norm(x)
            
        x = self.seq(x)
        x = self.dropout(F.gelu(x))
        
        if self.glu:
            x = self.out(x) * torch.sigmoid(self.out2(x))
        else:
            x = self.out(x)
            
        x = skip + self.dropout(x)
        
        if not self.prenorm:
            x = self.norm(x)
            
        return x


class Embedding(nn.Module):
    """Custom embedding layer"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, 1]
        y = self.embed(x[:, :, 0].long())
        return torch.where(x > 0, y, torch.zeros_like(y))


class StackedModel(nn.Module):
    """
    Complete sequence model with stacked SSM layers
    """
    def __init__(
        self,
        d_output,
        d_model,
        n_layers,
        layer_N,
        l_max,
        prenorm=True,
        dropout=0.0,
        embedding=False,
        classification=False,
        decode=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.classification = classification
        self.embedding = embedding
        self.decode = decode
        
        # Input encoder
        if embedding:
            self.encoder = Embedding(d_output, d_model)
        else:
            self.encoder = nn.Linear(1, d_model)
            
        # Stack of sequence processing layers
        self.layers = nn.ModuleList([
            SequenceBlock(
                d_model=d_model,
                N=layer_N,
                l_max=l_max,
                dropout=dropout,
                prenorm=prenorm,
                glu=True,
                decode=decode,
            )
            for _ in range(n_layers)
        ])
        
        # Output decoder
        self.decoder = nn.Linear(d_model, d_output)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, 1]
        
        # Transpose to [seq_len, batch_size, 1] for sequence processing
        x = x.transpose(0, 1)
        
        if not self.classification:
            if not self.embedding:
                # Normalize pixel values for image data
                x = x / 255.0
            if not self.decode:
                # Shift input for next-token prediction
                padding = torch.zeros_like(x[0:1])
                x = torch.cat([padding, x[:-1]], dim=0)
        
        # Encode input
        if self.embedding:
            x = self.encoder(x)
        else:
            # Ensure x is the right shape for the linear layer
            x = self.encoder(x)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x)
        
        if self.classification:
            # Global average pooling for classification
            x = torch.mean(x, dim=0)  # [batch_size, d_model]
            # Decode to output dimension
            x = self.decoder(x)
        else:
            # Transpose back to [batch_size, seq_len, d_model]
            x = x.transpose(0, 1)
            # Decode to output dimension
            x = self.decoder(x)
        
        # Apply log softmax for probability distribution
        return F.log_softmax(x, dim=-1)


# =============================================================================
# ========================= Train and Evaluate ================================
# =============================================================================

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_optimizer(model, lr, weight_decay=0.0):
    """Create AdamW optimizer for the model"""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, total_steps, lr_schedule=False):
    """Create learning rate scheduler if requested"""
    if lr_schedule:
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
    return None

def train_step(model, optimizer, scheduler, inputs, labels, classification=False):
    """Perform one training step"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(inputs)
    
    # Compute loss
    if not classification:
        # For sequence modeling, labels are the input pixels
        labels = inputs[:, :, 0].long()
        # For sequence modeling, logits are [batch_size, seq_len, n_classes]
        loss = F.nll_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
    else:
        # For classification, logits are [batch_size, n_classes]
        loss = F.nll_loss(logits, labels)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    return loss.item(), acc.item()

def eval_step(model, inputs, labels, classification=False):
    """Perform one evaluation step"""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        if not classification:
            # For sequence modeling, labels are the input pixels
            labels = inputs[:, :, 0].long()
            # For sequence modeling, logits are [batch_size, seq_len, n_classes]
            loss = F.nll_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
        else:
            # For classification, logits are [batch_size, n_classes]
            loss = F.nll_loss(logits, labels)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
    
    return loss.item(), acc.item()

def train_epoch(model, optimizer, scheduler, trainloader, classification=False, device='cuda'):
    """Train for one epoch"""
    batch_losses, batch_accuracies = [], []
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        loss, acc = train_step(model, optimizer, scheduler, inputs, labels, classification)
        batch_losses.append(loss)
        batch_accuracies.append(acc)
    
    return np.mean(batch_losses), np.mean(batch_accuracies)

def validate(model, testloader, classification=False, device='cuda'):
    """Validate the model"""
    losses, accuracies = [], []
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        loss, acc = eval_step(model, inputs, labels, classification)
        losses.append(loss)
        accuracies.append(acc)
    
    return np.mean(losses), np.mean(accuracies)

def main(case="classification"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration for the two cases
    if case == "classification":
        config = {
            "dataset": "mnist-classification",
            "layer": "s4",
            "seed": 1,
            "model": {
                "d_model": 8,    # Reduced from 128
                "n_layers": 2,    # Reduced from 4
                "dropout": 0.25,
                "prenorm": True,
                "embedding": False,
                "layer_N": 4     # Reduced from 64
            },
            "train": {
                "epochs": 5,      # Reduced from 20
                "bsz": 64,        # Reduced from 128
                "lr": 0.005,
                "lr_schedule": True,
                "weight_decay": 0.01
            }
        }
    else:
        config = {
            "dataset": "mnist",
            "layer": "s4",
            "seed": 0,
            "model": {
                "d_model": 8,    # Reduced from 128
                "n_layers": 2,    # Reduced from 4
                "dropout": 0.0,
                "prenorm": True,
                "embedding": False,
                "layer_N": 4     # Reduced from 64
            },
            "train": {
                "epochs": 5,      # Reduced from 100
                "bsz": 64,        # Reduced from 128
                "lr": 0.001,
                "lr_schedule": False,
                "weight_decay": 0.01
            }
        }

    # Set randomness
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    # Setup dataset
    classification = "classification" in config["dataset"]
    create_dataset_fn = Datasets[config["dataset"]]
    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(
        bsz=config["train"]["bsz"]
    )

    # Create model
    model = StackedModel(
        d_output=n_classes,
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        layer_N=config["model"]["layer_N"],
        l_max=l_max,
        prenorm=config["model"]["prenorm"],
        dropout=config["model"]["dropout"],
        embedding=config["model"]["embedding"],
        classification=classification,
    ).to(device)
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"[*] Trainable Parameters: {n_params}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model, 
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    total_steps = len(trainloader) * config["train"]["epochs"]
    print(f"[*] Total training steps: {total_steps}")
    
    scheduler = create_scheduler(
        optimizer, 
        total_steps=total_steps,
        lr_schedule=config["train"]["lr_schedule"]
    )

    # Training loop
    best_loss, best_acc, best_epoch = float('inf'), 0, 0
    
    for epoch in range(config["train"]["epochs"]):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        train_loss, train_acc = train_epoch(
            model, optimizer, scheduler, trainloader, 
            classification=classification, device=device
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(
            model, testloader, classification=classification, device=device
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy: {train_acc:.4f}")
        print(f"\tTest Loss: {test_loss:.5f} -- Test Accuracy: {test_acc:.4f}")

        if (classification and test_acc > best_acc) or (
            not classification and test_loss < best_loss
        ):
            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch
            # Save best model
            torch.save(model.state_dict(), f"best_model_{case}.pt")

        print(
            f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

if __name__ == "__main__":
    # Run classification case
    print("Running MNIST Classification case...")
    main("classification")
    
    # Run standard case
    print("\nRunning standard MNIST case...")
    main("standard")