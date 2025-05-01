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
            cls._device = Metal.MTLCreateSystemDefaultDevice()
            if cls._device:
                cls._command_queue = cls._device.newCommandQueue()
                cls._load_mps_functions()
                cls._mps_initialized = True
            else:
                raise RuntimeError("Metal device not found.")

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

    def to_cpu(self):
        if self._on_mps:
            try:
                if self._mps_buffer_data:
                    contents = np.frombuffer(self._mps_buffer_data.contents().as_buffer(self._mps_buffer_data.length()), dtype=np.float32)
                    self.data[:] = contents[:]
                if self._mps_buffer_grad:
                    contents = np.frombuffer(self._mps_buffer_grad.contents().as_buffer(self._mps_buffer_grad.length()), dtype=np.float32)
                    self.grad[:] = contents[:]
            except Exception as e:
                print(f"Error moving to CPU: {e}")
            finally:
                self._mps_buffer_data = None
                self._mps_buffer_grad = None
                self._on_mps = False

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
                out.to_cpu()
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
                out.to_cpu()
            except Exception as e:
                print(f"Error during MPS multiplication: {e}")
        def _backward():
            self.grad += other.data[0] * out.grad
            other.grad += self.data[0] * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data[0] ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data[0] ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
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
            self.grad += out.data * out.grad
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
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        x = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Function to update parameters on MPS using a Metal kernel
def update_parameters(parameters, learning_rate_np, pipeline_state):
    # Allocate shared debug buffers if needed (currently optional)
    for p in parameters:
        # Move parameter to MPS if not already there
        p.to_mps()
        command_buffer = Value._command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(p._mps_buffer_data, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(p._mps_buffer_grad, 0, 1)
        buffer_size = np.array([p.data.size], dtype=np.uint32)
        compute_encoder.setBytes_length_atIndex_(learning_rate_np.tobytes(), learning_rate_np.nbytes, 2)
        compute_encoder.setBytes_length_atIndex_(buffer_size.tobytes(), buffer_size.nbytes, 3)
        # Optional debug buffers could be set here at index 4 and 5
        grid_size = Metal.MTLSize(p.data.size, 1, 1)
        threadgroup_size = Metal.MTLSize(1, 1, 1)
        compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        p.to_cpu()


def update_parameters_mps(parameters, learning_rate_np, pipeline_state):
    # Flatten all parameters and gradients and reuse the buffers
    param_data = np.concatenate([p.data for p in parameters]).astype(np.float32)
    grad_data = np.concatenate([p.grad for p in parameters]).astype(np.float32)

    # Pre-allocate buffers and update only when necessary
    param_buffer = Value._device.newBufferWithBytes_length_options_(
        param_data.tobytes(), param_data.nbytes, Metal.MTLResourceStorageModeShared
    )
    grad_buffer = Value._device.newBufferWithBytes_length_options_(
        grad_data.tobytes(), grad_data.nbytes, Metal.MTLResourceStorageModeShared
    )
    
    # Command buffer and compute encoder setup
    command_buffer = Value._command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()
    compute_encoder.setComputePipelineState_(pipeline_state)
    
    # Set buffers for parameters and gradients
    compute_encoder.setBuffer_offset_atIndex_(param_buffer, 0, 0)
    compute_encoder.setBuffer_offset_atIndex_(grad_buffer, 0, 1)
    
    # Use buffer size as a single scalar
    buffer_size = np.array([param_data.size], dtype=np.uint32)
    compute_encoder.setBytes_length_atIndex_(learning_rate_np.tobytes(), learning_rate_np.nbytes, 2)
    compute_encoder.setBytes_length_atIndex_(buffer_size.tobytes(), buffer_size.nbytes, 3)
    
    # Optimizing grid and threadgroup sizes based on parameter count
    num_threads = param_data.size
    threads_per_group = 32  # This can be adjusted based on experimentation
    
    # If parameters are large, experiment with more efficient grid size
    grid_size = Metal.MTLSize(num_threads, 1, 1)
    threadgroup_size = Metal.MTLSize(threads_per_group, 1, 1)
    
    compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    compute_encoder.endEncoding()
    
    # Commit and wait for the command buffer to complete
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    # Efficiently read updated data from GPU buffer
    updated_param_data = np.frombuffer(param_buffer.contents().as_buffer(param_buffer.length()), dtype=np.float32)
    
    # Update parameters on CPU using batch operation
    offset = 0
    for p in parameters:
        size = p.data.size
        p.data[:] = updated_param_data[offset:offset + size]
        offset += size


# Training loop
if __name__ == '__main__':
    try:
        Value._initialize_mps()
        print(f"Using Metal device: {Value._device.name()}")
        mps_available = True
    except Exception as e:
        print(f"Error during Metal initialization: {e}")
        mps_available = False

        

    # Input and target data
    inputs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    targets = [1.0, -1.0, 0.5, -0.5]

    inputs_mps = [[Value(xi) for xi in x] for x in inputs]
    targets_mps = [Value(y) for y in targets]

    # Create a MLP with 3 inputs, two hidden layers of 4 neurons each, and 1 output
    network = MLP(3, [64, 64, 1])

    # Metal kernel for parameter updates
    update_kernel_source = """
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void update_params(device float *param [[buffer(0)]],
                              device float *grad [[buffer(1)]],
                              constant float &learning_rate [[buffer(2)]],
                              constant uint &buffer_size [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
        if (id < buffer_size) {
            param[id] -= learning_rate * grad[id];
        }
    }
    """

    pipeline_state = None
    if mps_available:
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

    learning_rate_np = np.array([0.01], dtype=np.float32)

    # Training loop
    use_mps = mps_available  # Toggle between CPU and MPS
    for epoch in range(10):
        total_loss = 0.0
        # Reset gradients
        for p in network.parameters():
            p.grad = np.array([0.0], dtype=np.float32)

        for x, y in zip(inputs_mps, targets_mps):
            # Forward pass
            pred = network(x)
            # Mean Squared Error loss
            loss = (pred - y) ** 2
            total_loss += loss.data[0]
            # Backward pass
            loss.backward()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

        # Update parameters
        if use_mps and pipeline_state:
            update_parameters_mps(network.parameters(), learning_rate_np, pipeline_state)
        else:
            # CPU-based parameter update
            for p in network.parameters():
                p.data -= learning_rate_np * p.grad