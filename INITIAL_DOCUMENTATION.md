\# AXON v1.0 Architecture Specification (Phase 1)



Version: 1.0

Language: C++20

Platform: Linux x86-64

Build: CMake

Backend: CPU only

Target: Train ResNet-18 on CIFAR/MNIST



\---



\# 1. Goals



Axon is a minimal deep learning framework focused on understanding how modern AI runtimes are built.



Phase 1 objectives:



\- CPU execution only

\- Eager execution with lightweight graph recording

\- Automatic differentiation

\- Train ResNet18 on ImageNet

\- GGML-inspired low precision tensor storage

\- Clean architecture suitable for future compiler/backend work



Not goals:



\- CUDA

\- Distributed training

\- JIT

\- Graph optimization

\- Compiler IR

\- Dynamic shape optimization



\---



\# 2. Architecture



```

Application

&#x20;     │

&#x20;     ▼

Neural Network API

&#x20;     │

&#x20;     ▼

Autograd

&#x20;     │

&#x20;     ▼

Runtime

&#x20;     │

&#x20;     ▼

CPU Backend

&#x20;     │

&#x20;     ▼

Storage

```



Only six major subsystems exist.



\---



\# 3. Repository



```

axon/



&#x20;   include/



&#x20;   src/



&#x20;       tensor/

&#x20;       storage/

&#x20;       runtime/

&#x20;       graph/

&#x20;       autograd/

&#x20;       nn/

&#x20;       optimizer/

&#x20;       backend/

&#x20;       utils/



&#x20;   tests/



&#x20;   examples/



&#x20;   third\_party/



&#x20;   CMakeLists.txt

```



\---



\# 4. Tensor



Tensor is a lightweight frontend object.



Tensor never owns memory.



```

Tensor



id



TensorType



Storage\*



requires\_grad

```



Tensor has no knowledge of graph topology.



Tensor contains no backward logic.



Tensor contains no kernel logic.



\---



\# 5. TensorType



TensorType describes data.



```

Shape



Stride



DType



Device



Quantization

```



TensorType is immutable.



\---



\# 6. Storage



Storage owns memory.



```

Storage



void\* data



size\_bytes



alignment



ref\_count



QuantizationDescriptor

```



Storage is reference-counted.



Multiple tensors may reference one storage.



Views share storage.



Clone allocates new storage.



\---



\# 7. Quantization



Quantization belongs to Storage.



Not Tensor.



Supported formats



```

None



Q8\_0



Q6\_K



Q5\_K



Q4\_0



Q4\_K



Q3\_K



Q2\_K

```



Each format defines



\- block size

\- packing

\- scales

\- decode rule



Runtime never understands packing.



Only backend kernels do.



\---



\# 8. Operations



Every operation implements



```

forward()



backward()

```



Examples



Add



Sub



Mul



Div



MatMul



Conv2D



ReLU



GELU



Softmax



LayerNorm



CrossEntropy



Reshape



Transpose



Slice



Mean



Sum



Max



\---



\# 9. Graph



Graph is only used for autograd.



```

Node



operation



inputs



outputs

```



During forward



each operation appends one node.



Backward traverses nodes in reverse order.



No optimization.



No scheduler.



No compiler.



\---



\# 10. Runtime



Runtime executes operations immediately.



Pseudo



```

forward()



↓



allocate outputs



↓



call backend



↓



record graph

```



No execution plan.



No VM.



No instruction set.



\---



\# 11. Backend



CPU only.



Backend implements numerical kernels.



```

add()



matmul()



conv2d()



relu()



softmax()



layernorm()

```



Backend receives



Tensor\*



Backend never accesses Graph.



\---



\# 12. Autograd



Autograd stores



```

Graph



Gradient Map

```



Backward



```

loss.backward()



↓



reverse graph



↓



operation.backward()



↓



accumulate gradients

```



No tape optimization.



\---



\# 13. Parameter



Parameter wraps Tensor.



```

Tensor



gradient



trainable

```



Nothing else.



\---



\# 14. Optimizer



Interface



```

step()



zero\_grad()

```



Implement



SGD



Adam



AdamW



\---



\# 15. Module



Base class



```

forward()



parameters()



train()



eval()

```



Modules



Linear



Conv2D



BatchNorm



LayerNorm



Embedding



Sequential



Residual



Flatten



Dropout



\---



\# 16. Losses



CrossEntropy



MSE



L1



NLL



\---



\# 17. Dataset



Interface



```

size()



get(index)

```



DataLoader



batching



shuffle



workers=0 (Phase 1)



\---



\# 18. Serialization



Support



Native binary



GGUF tensor import



Checkpoint



No ONNX.



\---



\# 19. Memory Rules



Tensor never owns memory.



Storage owns memory.



Graph owns nodes.



Module owns parameters.



Optimizer owns optimizer state.



\---



\# 20. Coding Rules



C++20 only.



No exceptions across subsystem boundaries.



RAII everywhere.



Raw pointers only for non-owning references.



Ownership uses



std::unique\_ptr



std::shared\_ptr



No macros.



No global state.



No singleton.



\---



\# 21. Milestones



M1



Tensor



Storage



MatMul



ReLU



Autograd



M2



Linear



SGD



MNIST



M3



Conv2D



AdamW



CIFAR10



M4



ResNet18



ImageNet



M5



GGML quantized inference/training research



\---



\# 22. Acceptance Criteria



Framework is complete when



\- Tensor abstraction is stable.

\- Autograd passes unit tests.

\- ResNet18 trains on ImageNet.

\- GGML-style quantized storage works.

\- CPU backend is deterministic.

\- Architecture allows future GPU backend without Tensor redesign.

