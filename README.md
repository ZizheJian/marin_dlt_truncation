# marin_dlt_truncation

**English** | [中文](README_CN.md)

`marin_dlt_truncation` is a C++17 header-only implementation of the lossy compressor for spectral-element data proposed by Marin, Schanen and Fischer:

> O. Marin, M. Schanen, P. Fischer. *Large-scale lossy data compression based on an a priori error estimator in a spectral element code.* Argonne National Laboratory, ANL/MCS-P6024-0616, 2016.

The original implementation lives inside Nek5000 and is not publicly available as a standalone library. This repository re-implements the method from the paper so that it can be used as a baseline.

## Method

Each element holds values on a tensor-product grid of Gauss–Lobatto–Legendre (GLL) nodes. The compressor processes every element independently:

1. **Transform.** It applies the discrete Legendre transform (DLT) along each dimension. The basis functions are Legendre polynomials scaled so that they are orthonormal under GLL quadrature (Eqs. 5 and 7–9 of the paper). Because of this scaling, the sum of squared coefficients equals the squared discrete L2 norm of the values on the reference element.
2. **Truncate.** It sorts the coefficients by magnitude and sets the smallest ones to zero, one at a time, as long as the error estimate stays below the user threshold `epsilon` (Section 4 of the paper, and Algorithm 1 of Otero et al., Flow Turbulence Combust. 101, 2018). The estimate is the volume-averaged L2 error of the element, `sqrt(sum of v_k^2 J_k over zeroed k / sum of w_i J_i over all nodes)`, where `w` are the GLL weights and `J` is the Jacobian of the mapping from the reference element. Because the transform is orthonormal, no inverse transform is needed to know the error.
3. **Encode.** Each zero coefficient is written as a single 0 bit. Every other coefficient is written as a 1 bit followed by all `m` bits of its floating-point value (Section 5 of the paper). No further lossless coding is applied.

Decompression decodes the coefficients and applies the inverse DLT.

When the volume-averaged error of every element is below `epsilon`, the volume-averaged error of the whole field is also below `epsilon`. The method does not bound the pointwise error, so individual values can deviate from the original by more than `epsilon`.

### Choices made where the paper leaves room

- **Truncation rule.** Algorithm 1 of the paper writes the truncation as `u_trunc = u_T > epsilon`, which reads like a per-coefficient threshold. Section 4 and Otero et al. describe a cumulative error criterion. This implementation follows the cumulative criterion.
- **Jacobian.** When `J` is constant within an element, it cancels and the estimate equals the true error. For curved elements, Eq. 13 of the paper multiplies each coefficient by the Jacobian at the node with the same index. This implementation does the same, using the normalization of KTH's NekDCTB (`trunc_field_l2oferror`), so that the estimate reduces to the exact error when `J` is constant. Passing no Jacobian treats it as constant.
- **Not included.** The paper also maps data to Chebyshev nodes and uses the discrete Chebyshev transform. It describes compressing on Nek5000's I/O ranks as well. Neither is part of this library.

## Usage

```cmake
add_subdirectory(third_party/marin_dlt_truncation/c_codes ${CMAKE_BINARY_DIR}/dlt_truncation_build)
target_link_libraries(my_target PRIVATE marin_dlt_truncation)
```

```cpp
#include <marin_dlt_truncation.hpp>

// An element is a C-order array of shape [n0,n1,n2]. Use n0=1 for 2D elements.
marin_dlt_truncation::element_transform<float> transform({8,8,8});
std::vector<float> coefficients(transform.size()),temp(transform.size());
std::vector<uint32_t> order;

// Optional: Jacobian from the physical coordinates of the nodes, one array per coordinate component
std::vector<float> jacobian(transform.size());
transform.jacobian({x,y,z},jacobian.data());

// Compression
std::vector<unsigned char> bytes;
marin_dlt_truncation::bit_writer writer(bytes);
for (each element)
{
    transform.forward(values,coefficients.data(),temp.data());
    marin_dlt_truncation::truncate(coefficients.data(),jacobian.data(),transform,epsilon,order); // or nullptr for constant J
    marin_dlt_truncation::encode<float>(coefficients.data(),transform.size(),writer);
}
writer.finish();

// Decompression
marin_dlt_truncation::bit_reader reader(bytes.data(),bytes.size());
for (each element)
{
    marin_dlt_truncation::decode<float>(reader,coefficients.data(),transform.size());
    transform.inverse(coefficients.data(),values,temp.data());
}
```

`truncate` returns the number of coefficients set to zero. The template argument of `encode` and `decode` is the stored floating-point type and decides `m`: 32 bits for `float` and 64 bits for `double`. It may differ from the type used for computation.

## Tests

```bash
cd tests
g++ -std=c++17 -O2 -Wall -I ../c_codes test_marin_dlt_truncation.cpp -o test_marin_dlt_truncation
./test_marin_dlt_truncation
```

The tests check the GLL nodes, that the transform is invertible and orthonormal, that the truncation error equals the estimate and stays below `epsilon`, that the Jacobian of an affine element is constant, and that encoding round-trips.
