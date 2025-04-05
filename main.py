import math
import random
import numpy as np
import objc
from Foundation import NSBundle
import Metal

class Value:
    _mps_initialized = False
    _device = None
    _command_queue = None
    _add_function = None
    _mul_function = None
    _tanh_function = None
    _exp_function = None

    def __init__(self, data, _children=(), _op="", label=''):
        self.data = np.array([float(data)], dtype=np.float32)
        self.grad = np.array([0.0], dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._on_mps = False
        self._mps_buffer_data = None
        self._mps_buffer_grad = None

    def __repr__(self):
        return f"Value(data={self.data[0]})"

    @classmethod
    def _initialize_mps(cls):
        if not cls._mps_initialized:
            print("Initializing Metal Performance Shaders (MPS)...")
            try:
                cls._device = Metal.MTLCreateSystemDefaultDevice()
                if cls._device:
                    cls._command_queue = cls._device.newCommandQueue()
                    cls._load_mps_functions()
                    cls._mps_initialized = True
                else:
                    raise RuntimeError("Metal device not found.")
            except Exception as e:
                print(f"Error initializing Metal: {e}")

    @classmethod
    def _load_mps_functions(cls):
        if cls._device and not cls._add_function:
            metal_source = """
                #include <metal_stdlib>
                using namespace metal;

                kernel void add_arrays(device const float *a [[buffer(0)]],
                                      device const float *b [[buffer(1)]],
                                      device float *result [[buffer(2)]],
                                      uint id [[thread_position_in_grid]]) {
                    result[id] = a[id] + b[id];
                }

                kernel void mul_arrays(device const float *a [[buffer(0)]],
                                      device const float *b [[buffer(1)]],
                                      device float *result [[buffer(2)]],
                                      uint id [[thread_position_in_grid]]) {
                    result[id] = a[id] * b[id];
                }

                kernel void tanh_array(device const float *a [[buffer(0)]],
                                      device float *result [[buffer(1)]],
                                      uint id [[thread_position_in_grid]]) {
                    result[id] = tanh(a[id]);
                }

                kernel void exp_array(device const float *a [[buffer(0)]],
                                    device float *result [[buffer(1)]],
                                    uint id [[thread_position_in_grid]]) {
                    result[id] = exp(a[id]);
                }
            """
            library, error = cls._device.newLibraryWithSource_options_error_(metal_source, None, None)
            if error is not None:
                raise RuntimeError(f"Failed to create Metal library: {error.localizedDescription() if hasattr(error, 'localizedDescription') else error}")
            if library:
                cls._add_function = library.newFunctionWithName_("add_arrays")
                cls._mul_function = library.newFunctionWithName_("mul_arrays")
                cls._tanh_function = library.newFunctionWithName_("tanh_array")
                cls._exp_function = library.newFunctionWithName_("exp_array")
            else:
                raise RuntimeError("Failed to create Metal library: Unknown error")

    def to_mps(self):
        if not self._on_mps:
            try:
                Value._initialize_mps()
                if Value._device:
                    self._mps_buffer_data = Value._device.newBufferWithBytes_length_options_(
                        self.data.tobytes(), self.data.nbytes, Metal.MTLResourceStorageModeShared
                    )
                    self._mps_buffer_grad = Value._device.newBufferWithBytes_length_options_(
                        self.grad.tobytes(), self.grad.nbytes, Metal.MTLResourceStorageModeShared
                    )
                    self._on_mps = True
                else:
                    print("Metal device not available.")
            except RuntimeError as e:
                print(f"Error moving to MPS: {e}")
                # Fallback if needed


    def to_cpu(self):
        if self._on_mps:
            try:
                if self._mps_buffer_data:
                    contents = np.frombuffer(self._mps_buffer_data.contents().as_buffer(self._mps_buffer_data.length()), dtype=np.float32)
                    self.data[:] = contents[:]
                if self._mps_buffer_grad:
                    contents = np.frombuffer(self._mps_buffer_grad.contents().as_buffer(self._mps_buffer_grad.length()), dtype=np.float32)
                    self.grad[:] = contents[:]
                self._mps_buffer_data = None
                self._mps_buffer_grad = None
                self._on_mps = False
            except Exception as e:
                print(f"Error moving to CPU: {e}")

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data[0] + other.data[0], (self, other), "+")

        if self._on_mps and other._on_mps and Value._device and Value._add_function:
            try:
                out.to_mps()
                command_buffer = Value._command_queue.commandBuffer()
                compute_encoder = command_buffer.computeCommandEncoder()
                compute_encoder.setComputePipelineState_(Value._add_function)
                compute_encoder.setBuffer_offset_atIndex_(self._mps_buffer_data, 0, 0)
                compute_encoder.setBuffer_offset_atIndex_(other._mps_buffer_data, 0, 1)
                compute_encoder.setBuffer_offset_atIndex_(out._mps_buffer_data, 0, 2)

                grid_size = Metal.MTLSize(1, 1, 1)
                threadgroup_size = Metal.MTLSize(1, 1, 1)
                compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

                compute_encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
                out.to_cpu() # Bring result back to CPU for now
            except Exception as e:
                print(f"Error during MPS addition: {e}")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data[0] * other.data[0], (self, other), "*")

        if self._on_mps and other._on_mps and Value._device and Value._mul_function:
            try:
                out.to_mps()
                command_buffer = Value._command_queue.commandBuffer()
                compute_encoder = command_buffer.computeCommandEncoder()
                compute_encoder.setComputePipelineState_(Value._mul_function)
                compute_encoder.setBuffer_offset_atIndex_(self._mps_buffer_data, 0, 0)
                compute_encoder.setBuffer_offset_atIndex_(other._mps_buffer_data, 0, 1)
                compute_encoder.setBuffer_offset_atIndex_(out._mps_buffer_data, 0, 2)

                grid_size = Metal.MTLSize(1, 1, 1)
                threadgroup_size = Metal.MTLSize(1, 1, 1)
                compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

                compute_encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
                out.to_cpu() # Bring result back to CPU for now
            except Exception as e:
                print(f"Error during MPS multiplication: {e}")

        def _backward():
            self.grad += other.data[0] * out.grad
            other.grad += self.data[0] * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data[0]**other, (self,), f'**{other}')
        def _backward ():
            self.grad += other * (self.data[0] ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self * other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def tanh(self):
        out = Value(np.tanh(self.data[0]), (self,), 'tanh')
        if self._on_mps and Value._device and Value._tanh_function:
            try:
                out.to_mps()
                command_buffer = Value._command_queue.commandBuffer()
                compute_encoder = command_buffer.computeCommandEncoder()
                compute_encoder.setComputePipelineState_(Value._tanh_function)
                compute_encoder.setBuffer_offset_atIndex_(self._mps_buffer_data, 0, 0)
                compute_encoder.setBuffer_offset_atIndex_(out._mps_buffer_data, 0, 1)

                grid_size = Metal.MTLSize(1, 1, 1)
                threadgroup_size = Metal.MTLSize(1, 1, 1)
                compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

                compute_encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
                out.to_cpu()
            except Exception as e:
                print(f"Error during MPS tanh: {e}")

        def _backward():
            t = np.tanh(self.data[0])
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data[0]), (self,), 'exp')
        if self._on_mps and Value._device and Value._exp_function:
            try:
                out.to_mps()
                command_buffer = Value._command_queue.commandBuffer()
                compute_encoder = command_buffer.computeCommandEncoder()
                compute_encoder.setComputePipelineState_(Value._exp_function)
                compute_encoder.setBuffer_offset_atIndex_(self._mps_buffer_data, 0, 0)
                compute_encoder.setBuffer_offset_atIndex_(out._mps_buffer_data, 0, 1)

                grid_size = Metal.MTLSize(1, 1, 1)
                threadgroup_size = Metal.MTLSize(1, 1, 1)
                compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

                compute_encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
                out.to_cpu()
            except Exception as e:
                print(f"Error during MPS exp: {e}")

        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = np.array([1.0], dtype=np.float32)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

class Neuron:
    """A single artificial neuron with a specified number of inputs (nin)."""

    def __init__(self, nin):
        # Initialize weights randomly in the range [-1, 1] for each input
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # Weight parameters
        # Initialize the bias term as a random value in the same range
        self.b = Value(random.uniform(-1, 1))  # Bias parameter

    def __call__(self, x):
        # Ensure all input values are wrapped as Value objects
        x = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]

        # Compute the weighted sum: Σ(x_i * w_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply activation function (tanh) to introduce non-linearity
        out = act.tanh()
        return out

    def parameters(self):
        # Return the neuron’s parameters (weights and bias) as a list
        params = self.w + [self.b]
        return params

class Layer:
    """A layer consisting of multiple neurons, forming a transformation of the input."""

    def __init__(self, nin, nout):
        # Initialize the layer with `nout` neurons, each taking `nin` inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]  # List of neurons

    def __call__(self, x):
        # Compute outputs of all neurons by calling them with input `x`
        out = [n(x) for n in self.neurons]
        # If there is only one neuron in the layer, return its output directly
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Collect and return all parameters of the neurons in the layer
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """A Multi-Layer Perceptron (MLP) consisting of multiple layers."""

    def __init__(self, nin, nouts):
        # Define the size of each layer by concatenating input size with output sizes
        sz = [nin] + nouts  # Example: nin=3, nouts=[4,2] -> [3,4,2]
        # Create layers based on the sizes: layer[i] transforms sz[i] -> sz[i+1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Forward propagate input through each layer
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Gather and return all parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == '__main__':
    use_mps = False
    try:
        # Attempt to initialize Metal once at the beginning
        Value._initialize_mps()
        if Value._device:
            print(f"Using Metal device: {Value._device.name()}")
            use_mps = True
    except ImportError:
        print("PyObjC or Metal not available. Using CPU.")
    except RuntimeError as e:
        print(f"Error during Metal initialization: {e}")

    # Define input data and ground truth
    inputs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    targets = [1.0, -1.0, 0.5, -0.5]

    # Move inputs and targets to MPS
    inputs_mps = [[Value(xi) for xi in x] for x in inputs]
    targets_mps = [Value(y) for y in targets]

    # Create an MLP with 3 inputs, two hidden layers of 4 neurons each, and 1 output
    n = MLP(3, [128, 128, 1])



    # Define the Metal kernel for parameter updates
    update_kernel_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void update_params(device float *param [[buffer(0)]],
                            device float *grad [[buffer(1)]],
                            constant float &learning_rate [[buffer(2)]],
                            constant uint &buffer_size [[buffer(3)]],
                            device float *debug_learning_rate [[buffer(4)]],
                            device float *debug_grad [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
        if (id < buffer_size) {
            param[id] -= learning_rate * grad[id];
            debug_learning_rate[id] = learning_rate; // Store the learning rate for debugging
            debug_grad[id] = grad[id];               // Store the gradient for debugging
        }
    }
    """

    # Compile the Metal kernel and create the pipeline state
    try:
        library, error = Value._device.newLibraryWithSource_options_error_(update_kernel_source, None, None)
        if error is not None:
            raise RuntimeError(f"Failed to create Metal library for updates: {error.localizedDescription()}")
        update_function = library.newFunctionWithName_("update_params")
        pipeline_state, error = Value._device.newComputePipelineStateWithFunction_error_(update_function, None)
        if error is not None:
            raise RuntimeError(f"Failed to create compute pipeline state: {error.localizedDescription()}")
    except Exception as e:
        print(f"Error creating pipeline state: {e}")
        pipeline_state = None

    # Training loop
    learning_rate_np = np.array([0.001], dtype=np.float32)  # Example learning rate
    for epoch in range(50):  # Example: 10 epochs
        total_loss = 0.0

        # Forward pass and loss computation
        for x, y in zip(inputs_mps, targets_mps):
            # Forward pass
            pred = n(x)

            # Compute mean squared error loss
            loss = (pred - y) ** 2
            total_loss += loss.data[0]

            # Backward pass
            loss.backward()


        # Move all parameters to MPS
        for p in n.parameters():
            p.to_mps()
            p.grad = np.array([0.0], dtype=np.float32)

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

        # print("pipeline_state", pipeline_state)
        # for p in n.parameters():
        #     p.data -= learning_rate_np * p.grad

        # Check if gradients are being calculated correctly
        # for p in n.parameters():
        #     print(f"Parameter before update: {p.data}")
        #     print(f"Gradient before update: {p.grad}")

        # Allocate debug buffers for learning_rate and grad
        debug_learning_rate_buffer = Value._device.newBufferWithLength_options_(
            p.data.nbytes, Metal.MTLResourceStorageModeShared
        )
        debug_grad_buffer = Value._device.newBufferWithLength_options_(
            p.data.nbytes, Metal.MTLResourceStorageModeShared
        )

        # for p in n.parameters():
        #     print(f"Gradient on CPU before moving to MPS: {p.grad}")

        for p in n.parameters():
            if p._on_mps:
                # Debug: Print learning rate
                # print(f"Learning rate: {learning_rate_np[0]}")
                # print(f"Grad buffer contents after moving to MPS: {np.frombuffer(p._mps_buffer_grad.contents().as_buffer(p._mps_buffer_grad.length()), dtype=np.float32)}")

                # Dispatch the kernel
                command_buffer = Value._command_queue.commandBuffer()
                compute_encoder = command_buffer.computeCommandEncoder()
                compute_encoder.setComputePipelineState_(pipeline_state)
                compute_encoder.setBuffer_offset_atIndex_(p._mps_buffer_data, 0, 0)
                compute_encoder.setBuffer_offset_atIndex_(p._mps_buffer_grad, 0, 1)
                buffer_size = np.array([p.data.size], dtype=np.uint32)
                compute_encoder.setBytes_length_atIndex_(learning_rate_np.tobytes(), learning_rate_np.nbytes, 2)
                compute_encoder.setBytes_length_atIndex_(buffer_size.tobytes(), buffer_size.nbytes, 3)
                compute_encoder.setBuffer_offset_atIndex_(debug_learning_rate_buffer, 0, 4)
                compute_encoder.setBuffer_offset_atIndex_(debug_grad_buffer, 0, 5)
                grid_size = Metal.MTLSize(p.data.size, 1, 1)
                threadgroup_size = Metal.MTLSize(1, 1, 1)
                compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
                compute_encoder.endEncoding()

                # print(f"Param buffer contents before dispatch: {np.frombuffer(p._mps_buffer_data.contents().as_buffer(p._mps_buffer_data.length()), dtype=np.float32)}")
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
                p.to_cpu()
                # print(f"Param buffer contents after dispatch: {np.frombuffer(p._mps_buffer_data.contents().as_buffer(p._mps_buffer_data.length()), dtype=np.float32)}")

                # # Debug: Read and print the learning rate and gradient from the debug buffers
                # debug_learning_rate = np.frombuffer(debug_learning_rate_buffer.contents().as_buffer(debug_learning_rate_buffer.length()), dtype=np.float32)
                # debug_grad = np.frombuffer(debug_grad_buffer.contents().as_buffer(debug_grad_buffer.length()), dtype=np.float32)
                # print(f"Debug learning rate buffer contents: {debug_learning_rate}")
                # print(f"Debug grad buffer contents: {debug_grad}")