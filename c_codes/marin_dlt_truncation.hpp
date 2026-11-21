#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Marin, Schanen, Fischer. Large-scale lossy data compression based on an a priori error estimator in a spectral element code. ANL/MCS-P6024-0616, 2016.
// 对GLL节点上的谱元数据，逐element做离散Legendre变换（DLT），把绝对值最小的系数置零，直到L2误差估计将要达到阈值epsilon，再把0编码成1个bit。
// 误差估计只约束每个element按体积平均的L2误差，不约束单点误差
namespace marin_dlt_truncation
{
    // n个GLL节点及其积分权重，在[-1,1]上从小到大排列
    inline void gll_nodes_and_weights(uint32_t n,std::vector<double> &nodes,std::vector<double> &weights)
    {
        if (n<2)
            throw std::invalid_argument("GLL节点数至少为2");
        uint32_t N=n-1;
        auto legendre=[N](double x,double &value,double &derivative)
        {
            double previous=1,current=x;
            for (uint32_t k=2;k<=N;k++)
            {
                double next=((2*k-1)*x*current-(k-1)*previous)/k;
                previous=current;
                current=next;
            }
            value=N==0?1:current;
            derivative=N*(x*current-previous)/(x*x-1);
        };
        nodes.assign(n,0);
        weights.assign(n,0);
        nodes[0]=-1;
        nodes[N]=1;
        // 内部节点是P_N'的根，用Chebyshev-Gauss-Lobatto点作初值做Newton迭代，(1-x^2)P_N'(x)的导数为-N(N+1)P_N(x)
        for (uint32_t i=1;i<N;i++)
        {
            double x=-cos(M_PI*i/N);
            for (int iteration=0;iteration<100;iteration++)
            {
                double p,dp;
                legendre(x,p,dp);
                double f=(1-x*x)*dp,df=-double(N)*(N+1)*p;
                double step=f/df;
                x-=step;
                if (fabs(step)<1e-16)
                    break;
            }
            nodes[i]=x;
        }
        // 端点处|P_N|=1
        for (uint32_t i=0;i<n;i++)
        {
            double p=1,dp;
            if (i!=0 and i!=N)
                legendre(nodes[i],p,dp);
            weights[i]=2.0/(N*(N+1)*p*p);
        }
    }

    // 一个element上的DLT。element是按C顺序存放的三维数组，shape为[n0,n1,n2]，长度为1的维度不参与变换（二维element令n0=1）。
    // 第a维的基函数是归一化的Legendre多项式c_k*P_k，使离散正交关系sum_i w_i*c_k*P_k(x_i)*c_l*P_l(x_i)=delta_kl成立（Marin式(5)、(9)）。
    // 因此系数的平方和等于参考element上数值的离散L2范数的平方
    template<typename T>
    class element_transform
    {
        public:
            explicit element_transform(std::array<uint32_t,3> shape)
            {
                this->shape=shape;
                element_size=shape[0]*shape[1]*shape[2];
                node_weights.assign(element_size,1);
                for (int axis=0;axis<3;axis++)
                {
                    uint32_t n=shape[axis];
                    if (n==1)
                        continue;
                    active_axes.push_back(axis);
                    std::vector<double> x,w;
                    gll_nodes_and_weights(n,x,w);
                    // basis[i*n+k]=c_k*P_k(x_i)，逆变换u=basis*v；正变换v=basis^T*W*u
                    std::vector<double> basis(n*n);
                    for (uint32_t i=0;i<n;i++)
                    {
                        double previous=1,current=x[i];
                        for (uint32_t k=0;k<n;k++)
                        {
                            double p;
                            if (k==0)
                                p=1;
                            else if (k==1)
                                p=x[i];
                            else
                            {
                                double next=((2*k-1)*x[i]*current-(k-1)*previous)/k;
                                previous=current;
                                current=next;
                                p=current;
                            }
                            double norm=k<n-1?(2.0*k+1)/2:(n-1)/2.0;
                            basis[i*n+k]=sqrt(norm)*p;
                        }
                    }
                    inverse_matrix[axis].resize(n*n);
                    forward_matrix[axis].resize(n*n);
                    for (uint32_t i=0;i<n;i++)
                        for (uint32_t k=0;k<n;k++)
                        {
                            inverse_matrix[axis][i*n+k]=static_cast<T>(basis[i*n+k]);
                            forward_matrix[axis][k*n+i]=static_cast<T>(basis[i*n+k]*w[i]);
                        }
                    // 节点间求导矩阵D[i*n+j]=l_j'(x_i)，计算Jacobian时使用
                    derivative_matrix[axis].assign(n*n,0);
                    std::vector<double> pn(n);
                    for (uint32_t i=0;i<n;i++)
                        pn[i]=legendre_value(n-1,x[i]);
                    for (uint32_t i=0;i<n;i++)
                        for (uint32_t j=0;j<n;j++)
                            if (i!=j)
                                derivative_matrix[axis][i*n+j]=static_cast<T>(pn[i]/(pn[j]*(x[i]-x[j])));
                    derivative_matrix[axis][0]=static_cast<T>(-double(n-1)*n/4);
                    derivative_matrix[axis][n*n-1]=static_cast<T>(double(n-1)*n/4);
                    for (uint32_t index=0;index<element_size;index++)
                    {
                        uint32_t coordinate=(axis==0?index/(shape[1]*shape[2]):(axis==1?index/shape[2]%shape[1]:index%shape[2]));
                        node_weights[index]*=static_cast<T>(w[coordinate]);
                    }
                }
                reference_volume=std::accumulate(node_weights.begin(),node_weights.end(),T(0));
            }

            uint32_t size() const {return element_size;}
            uint32_t dimension() const {return active_axes.size();}
            // 各节点上的张量积GLL权重，和为参考element的体积2^dimension
            const std::vector<T> &weights() const {return node_weights;}
            T volume() const {return reference_volume;}

            // 数值到系数。temp是长度为size()的工作区
            void forward(const T *values,T *coefficients,T *temp) const
            {
                apply(forward_matrix,values,coefficients,temp);
            }

            // 系数到数值
            void inverse(const T *coefficients,T *values,T *temp) const
            {
                apply(inverse_matrix,coefficients,values,temp);
            }

            // 由各节点的物理坐标计算Jacobian行列式的绝对值。coordinates[d]是第d个坐标分量，分量数等于dimension()
            void jacobian(const std::vector<const T*> &coordinates,T *result) const
            {
                uint32_t d=dimension();
                if (coordinates.size()!=d)
                    throw std::invalid_argument("坐标分量数与element维数不一致");
                std::vector<T> gradient(d*d*element_size);
                // gradient[(c*d+a)*element_size+index]=第c个坐标分量沿第a个参考方向的导数
                for (uint32_t c=0;c<d;c++)
                    for (uint32_t a=0;a<d;a++)
                        derivative(active_axes[a],coordinates[c],gradient.data()+(c*d+a)*element_size);
                for (uint32_t index=0;index<element_size;index++)
                {
                    auto g=[&](uint32_t c,uint32_t a){return gradient[(c*d+a)*element_size+index];};
                    T determinant;
                    if (d==1)
                        determinant=g(0,0);
                    else if (d==2)
                        determinant=g(0,0)*g(1,1)-g(0,1)*g(1,0);
                    else
                        determinant=g(0,0)*(g(1,1)*g(2,2)-g(1,2)*g(2,1))-g(0,1)*(g(1,0)*g(2,2)-g(1,2)*g(2,0))+g(0,2)*(g(1,0)*g(2,1)-g(1,1)*g(2,0));
                    result[index]=fabs(determinant);
                }
            }

        private:
            std::array<uint32_t,3> shape;
            uint32_t element_size;
            std::vector<int> active_axes;
            std::array<std::vector<T>,3> forward_matrix,inverse_matrix,derivative_matrix;
            std::vector<T> node_weights;
            T reference_volume;

            // P_N(x)
            static double legendre_value(uint32_t N,double x)
            {
                double previous=1,current=x;
                for (uint32_t k=2;k<=N;k++)
                {
                    double next=((2*k-1)*x*current-(k-1)*previous)/k;
                    previous=current;
                    current=next;
                }
                return N==0?1:current;
            }

            // output沿axis维乘以matrix：output[...,i,...]=sum_j matrix[i*n+j]*input[...,j,...]
            void apply_axis(const std::vector<T> &matrix,int axis,const T *input,T *output) const
            {
                uint32_t n=shape[axis];
                uint32_t stride=axis==0?shape[1]*shape[2]:(axis==1?shape[2]:1);
                uint32_t outer=element_size/(n*stride);
                for (uint32_t o=0;o<outer;o++)
                    for (uint32_t s=0;s<stride;s++)
                    {
                        const T *in=input+o*n*stride+s;
                        T *out=output+o*n*stride+s;
                        for (uint32_t i=0;i<n;i++)
                        {
                            T sum=0;
                            for (uint32_t j=0;j<n;j++)
                                sum+=matrix[i*n+j]*in[j*stride];
                            out[i*stride]=sum;
                        }
                    }
            }

            void apply(const std::array<std::vector<T>,3> &matrices,const T *input,T *output,T *temp) const
            {
                if (active_axes.empty())
                {
                    std::copy(input,input+element_size,output);
                    return;
                }
                // 每次乘一个维度，在output和temp之间交替，保证最后一次写进output
                const T *source=input;
                T *buffers[2]={active_axes.size()%2==1?output:temp,active_axes.size()%2==1?temp:output};
                for (size_t a=0;a<active_axes.size();a++)
                {
                    apply_axis(matrices[active_axes[a]],active_axes[a],source,buffers[a%2]);
                    source=buffers[a%2];
                }
            }

            void derivative(int axis,const T *input,T *output) const
            {
                apply_axis(derivative_matrix[axis],axis,input,output);
            }
    };

    // Marin第4节、Otero et al. 2018 Algorithm 1的截断：系数按绝对值从小到大排序，依次置零，只要置零之后的误差估计仍小于epsilon就继续。返回置零的系数个数。
    // 误差估计是element按体积平均的L2误差：sqrt(sum_{置零的k} v_k^2*J_k/sum_i w_i*J_i)。jacobian为nullptr时按Jacobian为常数处理，估计值等于真实误差。
    // Jacobian不为常数时，Marin式(13)把节点上的Jacobian直接乘到同一位置的系数上，这里的处理与KTH NekDCTB（trunc_field_l2oferror）相同。
    // order是长度为element大小的工作区
    template<typename T>
    uint32_t truncate(T *coefficients,const T *jacobian,const element_transform<T> &transform,T epsilon,std::vector<uint32_t> &order)
    {
        uint32_t n=transform.size();
        order.resize(n);
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return fabs(coefficients[a])<fabs(coefficients[b]);});
        T volume=transform.volume();
        if (jacobian)
        {
            volume=0;
            for (uint32_t i=0;i<n;i++)
                volume+=transform.weights()[i]*jacobian[i];
        }
        T limit=epsilon*epsilon*volume,error_sum=0;
        uint32_t zero_num=0;
        for (uint32_t k:order)
        {
            T contribution=coefficients[k]*coefficients[k]*(jacobian?jacobian[k]:T(1));
            if (error_sum+contribution>=limit)
                break;
            error_sum+=contribution;
            coefficients[k]=0;
            zero_num++;
        }
        return zero_num;
    }

    // 按bit写入，低位在前
    class bit_writer
    {
        public:
            explicit bit_writer(std::vector<unsigned char> &output)
            {
                this->output=&output;
            }
            // 写入bits的低n位，n不超过32
            void write(uint32_t bits,int n)
            {
                buffer|=uint64_t(bits)<<count;
                count+=n;
                while (count>=8)
                {
                    output->push_back(static_cast<unsigned char>(buffer));
                    buffer>>=8;
                    count-=8;
                }
            }
            // 把不满一个字节的剩余bit补0写出
            void finish()
            {
                if (count>0)
                    output->push_back(static_cast<unsigned char>(buffer));
                buffer=0;
                count=0;
            }
        private:
            std::vector<unsigned char> *output; // 写入的目标，指向构造时传入的数组
            uint64_t buffer=0;
            int count=0;
    };

    class bit_reader
    {
        public:
            bit_reader(const unsigned char *data,size_t size)
            {
                this->data=data;
                this->size=size;
            }
            uint32_t read(int n)
            {
                while (count<n)
                {
                    if (position>=size)
                        throw std::runtime_error("marin_dlt_truncation bitstream不完整");
                    buffer|=uint64_t(data[position++])<<count;
                    count+=8;
                }
                uint32_t bits=static_cast<uint32_t>(buffer&((uint64_t(1)<<n)-1));
                buffer>>=n;
                count-=n;
                return bits;
            }
            size_t bytes_read() const {return position;}
        private:
            const unsigned char *data;
            size_t size;
            size_t position=0;
            uint64_t buffer=0;
            int count=0;
    };

    // Marin第5节的编码：值为0的系数写1个bit 0，其余系数写1个bit 1，再写它的全部m位（F为float时m=32，为double时m=64）
    template<typename F,typename T>
    void encode(const T *coefficients,uint32_t count,bit_writer &writer)
    {
        static_assert(sizeof(F)==4 or sizeof(F)==8);
        for (uint32_t i=0;i<count;i++)
        {
            F value=static_cast<F>(coefficients[i]);
            if (value==0)
            {
                writer.write(0,1);
                continue;
            }
            writer.write(1,1);
            if constexpr (sizeof(F)==4)
            {
                uint32_t bits;
                memcpy(&bits,&value,4);
                writer.write(bits,32);
            }
            else
            {
                uint64_t bits;
                memcpy(&bits,&value,8);
                writer.write(static_cast<uint32_t>(bits),32);
                writer.write(static_cast<uint32_t>(bits>>32),32);
            }
        }
    }

    template<typename F,typename T>
    void decode(bit_reader &reader,T *coefficients,uint32_t count)
    {
        static_assert(sizeof(F)==4 or sizeof(F)==8);
        for (uint32_t i=0;i<count;i++)
        {
            if (reader.read(1)==0)
            {
                coefficients[i]=0;
                continue;
            }
            F value;
            if constexpr (sizeof(F)==4)
            {
                uint32_t bits=reader.read(32);
                memcpy(&value,&bits,4);
            }
            else
            {
                uint64_t low=reader.read(32),high=reader.read(32);
                uint64_t bits=low|(high<<32);
                memcpy(&value,&bits,8);
            }
            coefficients[i]=static_cast<T>(value);
        }
    }
}
