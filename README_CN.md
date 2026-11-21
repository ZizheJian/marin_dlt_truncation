# marin_dlt_truncation

[English](README.md) | **中文**

`marin_dlt_truncation`是Marin、Schanen和Fischer提出的谱元数据有损压缩方法的C++17头文件实现：

> O. Marin, M. Schanen, P. Fischer. *Large-scale lossy data compression based on an a priori error estimator in a spectral element code.* Argonne National Laboratory, ANL/MCS-P6024-0616, 2016.

原实现在Nek5000内部，没有公开的独立代码。这个仓库按论文重新实现了这个方法，用作对比的baseline。

## 方法

每个element在张量积Gauss–Lobatto–Legendre（GLL）节点上存放数值。压缩器对每个element独立处理：

1. **变换。** 在每个维度上做离散Legendre变换（DLT）。基函数是经过缩放的Legendre多项式，在GLL积分下正交归一（论文式(5)、(7)–(9)）。因此系数的平方和等于参考element上数值的离散L2范数的平方。
2. **截断。** 把系数按绝对值从小到大排序，逐个置零，只要误差估计仍小于用户给定的阈值`epsilon`就继续（论文第4节，以及Otero et al., Flow Turbulence Combust. 101, 2018的Algorithm 1）。误差估计是element按体积平均的L2误差，即`sqrt(置零系数的v_k^2*J_k之和/所有节点的w_i*J_i之和)`，其中`w`是GLL权重，`J`是参考element到实际element的映射的Jacobian。由于变换正交归一，不需要做逆变换就能知道误差。
3. **编码。** 值为0的系数写1个bit 0，其余系数写1个bit 1，后面跟它的全部`m`位浮点数（论文第5节），不再做其他无损编码。

解压时解码出系数，再做逆DLT。

每个element按体积平均的误差都小于`epsilon`时，整个场按体积平均的误差也小于`epsilon`。这个方法不约束单点误差，个别数值与原值的差可能超过`epsilon`。

### 论文没有写死、需要选择的地方

- **截断规则。** 论文的Algorithm 1把截断写成`u_trunc = u_T > epsilon`，看上去像是对单个系数设阈值。论文第4节和Otero et al.描述的是按累积误差判断。这里按累积误差实现。
- **Jacobian。** element内`J`为常数时，它会被约掉，误差估计等于真实误差。对于弯曲的element，论文式(13)把同一编号节点上的Jacobian乘到对应系数上。这里也这样做，归一化方式与KTH的NekDCTB（`trunc_field_l2oferror`）相同，使`J`为常数时估计值等于真实误差。不传Jacobian时按常数处理。
- **没有实现的部分。** 论文还考虑了先映射到Chebyshev节点、再做离散Chebyshev变换的做法，以及在Nek5000的I/O进程上压缩。这两部分都不在这个库里。

## 用法

```cmake
add_subdirectory(third_party/marin_dlt_truncation/c_codes ${CMAKE_BINARY_DIR}/dlt_truncation_build)
target_link_libraries(my_target PRIVATE marin_dlt_truncation)
```

```cpp
#include <marin_dlt_truncation.hpp>

// element是按C顺序存放、shape为[n0,n1,n2]的数组，二维element令n0=1
marin_dlt_truncation::element_transform<float> transform({8,8,8});
std::vector<float> coefficients(transform.size()),temp(transform.size());
std::vector<uint32_t> order;

// 可选：由节点的物理坐标计算Jacobian，每个坐标分量一个数组
std::vector<float> jacobian(transform.size());
transform.jacobian({x,y,z},jacobian.data());

// 压缩
std::vector<unsigned char> bytes;
marin_dlt_truncation::bit_writer writer(bytes);
for (each element)
{
    transform.forward(values,coefficients.data(),temp.data());
    marin_dlt_truncation::truncate(coefficients.data(),jacobian.data(),transform,epsilon,order); // J按常数处理时传nullptr
    marin_dlt_truncation::encode<float>(coefficients.data(),transform.size(),writer);
}
writer.finish();

// 解压
marin_dlt_truncation::bit_reader reader(bytes.data(),bytes.size());
for (each element)
{
    marin_dlt_truncation::decode<float>(reader,coefficients.data(),transform.size());
    transform.inverse(coefficients.data(),values,temp.data());
}
```

`truncate`返回置零的系数个数。`encode`和`decode`的模板参数是存储用的浮点类型，决定`m`：`float`为32位，`double`为64位。它可以和计算用的类型不同。

## 测试

```bash
cd tests
g++ -std=c++17 -O2 -Wall -I ../c_codes test_marin_dlt_truncation.cpp -o test_marin_dlt_truncation
./test_marin_dlt_truncation
```

测试检查GLL节点，变换可逆且正交归一，截断误差等于误差估计并小于`epsilon`，仿射element的Jacobian为常数，以及编码解码一致。
