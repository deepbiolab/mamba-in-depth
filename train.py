import os
from tqdm import tqdm
from functools import partial

import torch
import torchvision
import torchvision.transforms as transforms

import jax
import jax.numpy as np
from jax.scipy.signal import convolve
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power

import optax
from flax import linen as nn
from flax.training import train_state

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
    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
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
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
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

def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C

def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C

def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)

def run_SSM(A, B, C, u):
    """
    Run a State Space Model (SSM).
    
    Parameters:
        A: Continuous-time state matrix (NxN)
        B: Continuous-time input matrix (Nx1)
        C: Output matrix (1xN)
        u: Input signal (L-dimensional vector)
    
    Returns:
        The output of the state space model after running.
    """
    L = u.shape[0]
    N = A.shape[0]
    
    # Discretize the continuous-time state space model (A, B, C)
    # step is the discretization step size, set to 1.0 / L here
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    
    # Run the recursive computation of the state space model
    # Initial state is set to zero vector np.zeros((N,))
    # scan_SSM returns two values, we only take the second one which is the model output
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]

def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )

def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]
    
def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init

class SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u
        
def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )
SSMLayer = cloneLayer(SSMLayer)

class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x
    
class Embedding(nn.Embed):
    num_embeddings: int     # Vocabulary size
    features: int           # Embedding dimension

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)

class StackedModel(nn.Module):
    layer_cls: nn.Module    # Layer class to use (e.g., SSMLayer)
    layer: dict             # Layer configuration parameters
    d_output: int           # Output dimension (e.g., vocab size)
    d_model: int            # Model hidden dimension
    n_layers: int           # Number of layers to stack
    prenorm: bool = True    # Whether to apply normalization before layer
    dropout: float = 0.0    # Dropout rate
    embedding: bool = False # Whether to use embedding or dense layer for encoding
    classification: bool = False
    training: bool = True
    decode: bool = False    # Decoding mode flag (for autoregressive generation)

    def setup(self):
        # Input encoder
        if self.embedding:
            # For discrete inputs (e.g., tokens)
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            # For continuous inputs
            self.encoder = nn.Dense(self.d_model)
        
        # Output decoder
        self.decoder = nn.Dense(self.d_output)
        
        # Stack of sequence processing layers
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                # Normalize pixel values for image data
                x = x / 255.0
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        
        # Encode input
        x = self.encoder(x)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x)
        
        if self.classification:
            # Global average pooling for classification
            x = np.mean(x, axis=0)
        
        # Decode to output dimension
        x = self.decoder(x)

        # Apply log softmax for probability distribution
        return nn.log_softmax(x, axis=-1)

BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)    

# =============================================================================
# ========================= Train and Evaluate ================================
# =============================================================================


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

def map_nested_fn(fn):
    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }
    return map_fn

def create_train_state(rng, model_cls, trainloader, lr, lr_layer=None, 
                      lr_schedule=False, weight_decay=0.0, total_steps=-1):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[0].numpy()),
    )
    params = params["params"].unfreeze()

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: lr
    
    if lr_layer is None:
        lr_layer = {}

    optimizers = {
        k: optax.adam(learning_rate=schedule_fn(v * lr))
        for k, v in lr_layer.items()
    }
    optimizers["__default__"] = optax.adamw(
        learning_rate=schedule_fn(lr),
        weight_decay=weight_decay,
    )
    
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)

    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if param.dtype in [np.complex64, np.complex128] else 1)
        if lr_layer.get(k, lr) > 0.0 else 0
    )(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@partial(jax.jit, static_argnums=(4, 5))
def train_step(state, rng, batch_inputs, batch_labels, model, classification=False):
    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        return loss, (logits, acc)

    if not classification:
        batch_labels = batch_inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if not classification:
        batch_labels = batch_inputs[:, :, 0]
    logits = model.apply({"params": params}, batch_inputs)
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc

def train_epoch(state, rng, model, trainloader, classification=False):
    model = model(training=True)
    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())
        rng, drop_rng = jax.random.split(rng)
        state, loss, acc = train_step(state, drop_rng, inputs, labels, model, classification)
        batch_losses.append(loss)
        batch_accuracies.append(acc)
    return state, np.mean(np.array(batch_losses)), np.mean(np.array(batch_accuracies))

def validate(params, model, testloader, classification=False):
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())
        loss, acc = eval_step(inputs, labels, params, model, classification)
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(np.array(losses)), np.mean(np.array(accuracies))

def main(case="classification"):
    # Configuration for the two cases
    if case == "classification":
        config = {
            "dataset": "mnist-classification",
            "layer": "s4",
            "seed": 1,
            "model": {
                "d_model": 128,
                "n_layers": 4,
                "dropout": 0.25,
                "prenorm": True,
                "embedding": False,
                "layer": {"N": 64}
            },
            "train": {
                "epochs": 20,
                "bsz": 128,
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
                "d_model": 128,
                "n_layers": 4,
                "dropout": 0.0,
                "prenorm": True,
                "embedding": False,
                "layer": {"N": 64}
            },
            "train": {
                "epochs": 100,
                "bsz": 128,
                "lr": 0.001,
                "lr_schedule": False,
                "weight_decay": 0.01
            }
        }

    # Set randomness
    torch.random.manual_seed(config["seed"])
    key = jax.random.PRNGKey(config["seed"])
    key, rng, train_rng = jax.random.split(key, num=3)

    # Setup dataset
    classification = "classification" in config["dataset"]
    create_dataset_fn = Datasets[config["dataset"]]
    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(
        bsz=config["train"]["bsz"]
    )

    # Setup model
    config["model"]["layer"]["l_max"] = l_max
    model_cls = partial(
        BatchStackedModel,
        layer_cls=SSMLayer,
        d_output=n_classes,
        classification=classification,
        **config["model"]
    )

    # Create training state
    state = create_train_state(
        rng,
        model_cls,
        trainloader,
        lr=config["train"]["lr"],
        lr_layer=getattr(SSMLayer, "lr", None),
        lr_schedule=config["train"]["lr_schedule"],
        weight_decay=config["train"]["weight_decay"],
        total_steps=len(trainloader) * config["train"]["epochs"],
    )

    # Training loop
    best_loss, best_acc, best_epoch = 10000, 0, 0
    for epoch in range(config["train"]["epochs"]):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss, train_acc = train_epoch(
            state, train_rng, model_cls, trainloader, classification=classification
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(
            state.params, model_cls, testloader, classification=classification
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy: {train_acc:.4f}")
        print(f"\tTest Loss: {test_loss:.5f} -- Test Accuracy: {test_acc:.4f}")

        if (classification and test_acc > best_acc) or (
            not classification and test_loss < best_loss
        ):
            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch

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