// 编译运行见README
#include <marin_dlt_truncation.hpp>
#include <cstdio>
#include <random>

using namespace marin_dlt_truncation;

static int failures=0;
static void check(bool condition,const char *message,double value=0)
{
    if (!condition)
    {
        printf("FAILED: %s (%g)\n",message,value);
        failures++;
    }
}

template<typename T>
void test_shape(std::array<uint32_t,3> shape,T tolerance)
{
    element_transform<T> transform(shape);
    uint32_t n=transform.size();
    std::mt19937 rng(1);
    std::normal_distribution<double> normal;
    std::vector<T> u(n),v(n),back(n),temp(n);
    for (auto &value:u)
        value=normal(rng);
    transform.forward(u.data(),v.data(),temp.data());
    transform.inverse(v.data(),back.data(),temp.data());
    double max_error=0,coefficient_energy=0,value_energy=0;
    for (uint32_t i=0;i<n;i++)
    {
        max_error=std::max(max_error,fabs(double(back[i])-u[i]));
        coefficient_energy+=double(v[i])*v[i];
        value_energy+=double(transform.weights()[i])*u[i]*u[i];
    }
    check(max_error<tolerance,"inverse(forward(u))!=u",max_error);
    // 系数平方和等于参考element上的离散L2范数平方
    check(fabs(coefficient_energy-value_energy)<tolerance*value_energy,"Parseval relation",coefficient_energy-value_energy);
    check(fabs(transform.volume()-std::pow(2.0,transform.dimension()))<tolerance,"reference volume",transform.volume());
    // 截断后的实际误差等于误差估计，并且小于epsilon
    std::vector<uint32_t> order;
    for (double epsilon:{1e-3,1e-1,0.5,10.0})
    {
        std::vector<T> truncated=v;
        uint32_t zero_num=truncate(truncated.data(),static_cast<const T*>(nullptr),transform,T(epsilon),order);
        transform.inverse(truncated.data(),back.data(),temp.data());
        double error_energy=0,estimate=0;
        for (uint32_t i=0;i<n;i++)
        {
            error_energy+=double(transform.weights()[i])*(back[i]-u[i])*(back[i]-u[i]);
            if (truncated[i]==0)
                estimate+=double(v[i])*v[i];
        }
        double error=sqrt(error_energy/transform.volume());
        check(error<epsilon*(1+10*tolerance),"error exceeds epsilon",error);
        check(fabs(error-sqrt(estimate/transform.volume()))<10*tolerance*std::max(1.0,error),"estimate differs from error",error-sqrt(estimate/transform.volume()));
        // 再多置零一个系数就会超过epsilon
        if (zero_num<n)
        {
            double next=1e300;
            for (uint32_t i=0;i<n;i++)
                if (truncated[i]!=0)
                    next=std::min(next,double(v[i])*v[i]);
            check(sqrt((estimate+next)/transform.volume())>=epsilon*(1-10*tolerance),"truncation stopped too early");
        }
    }
    // 仿射element的Jacobian是常数：x=A*r+b
    uint32_t d=transform.dimension();
    std::vector<std::vector<T>> coordinates(d,std::vector<T>(n));
    double A[3][3]={{0.7,0.2,0.1},{-0.3,1.5,0.4},{0.2,-0.1,0.9}};
    std::vector<int> axes;
    for (int axis=0;axis<3;axis++)
        if (shape[axis]>1)
            axes.push_back(axis);
    std::vector<std::vector<double>> nodes(3),weights(3);
    for (int axis:axes)
        gll_nodes_and_weights(shape[axis],nodes[axis],weights[axis]);
    for (uint32_t index=0;index<n;index++)
    {
        uint32_t c[3]={index/(shape[1]*shape[2]),index/shape[2]%shape[1],index%shape[2]};
        for (uint32_t row=0;row<d;row++)
        {
            double value=1.0+row;
            for (uint32_t a=0;a<d;a++)
                value+=A[row][a]*nodes[axes[a]][c[axes[a]]];
            coordinates[row][index]=static_cast<T>(value);
        }
    }
    std::vector<const T*> pointers;
    for (auto &coordinate:coordinates)
        pointers.push_back(coordinate.data());
    std::vector<T> jacobian(n);
    transform.jacobian(pointers,jacobian.data());
    double expected=d==1?A[0][0]:(d==2?A[0][0]*A[1][1]-A[0][1]*A[1][0]:A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]));
    double max_jacobian_error=0;
    for (auto value:jacobian)
        max_jacobian_error=std::max(max_jacobian_error,fabs(value-fabs(expected)));
    check(max_jacobian_error<100*tolerance,"affine Jacobian",max_jacobian_error);
    // Jacobian为常数时，传入Jacobian与不传结果相同
    std::vector<T> with_jacobian=v,without_jacobian=v;
    truncate(with_jacobian.data(),jacobian.data(),transform,T(0.1),order);
    truncate(without_jacobian.data(),static_cast<const T*>(nullptr),transform,T(0.1),order);
    uint32_t differences=0;
    for (uint32_t i=0;i<n;i++)
        differences+=(with_jacobian[i]==0)!=(without_jacobian[i]==0);
    check(differences<=1,"constant Jacobian changes truncation",differences);
    // 编码后解码得到相同的系数
    std::vector<unsigned char> bytes;
    bit_writer writer(bytes);
    encode<T>(with_jacobian.data(),n,writer);
    encode<T>(v.data(),n,writer);
    writer.finish();
    bit_reader reader(bytes.data(),bytes.size());
    std::vector<T> decoded(2*n);
    decode<T>(reader,decoded.data(),n);
    decode<T>(reader,decoded.data()+n,n);
    check(std::equal(with_jacobian.begin(),with_jacobian.end(),decoded.begin()) and std::equal(v.begin(),v.end(),decoded.begin()+n),"encode/decode round trip");
    uint32_t nonzero=n-std::count(with_jacobian.begin(),with_jacobian.end(),T(0));
    check(bytes.size()==(2*n+(nonzero+n)*8*sizeof(T)+7)/8,"encoded size",bytes.size());
    printf("shape=[%u,%u,%u] %s: max round-trip error=%g, Parseval difference=%g\n",shape[0],shape[1],shape[2],sizeof(T)==4?"float":"double",max_error,coefficient_energy-value_energy);
}

int main()
{
    std::vector<double> x,w;
    gll_nodes_and_weights(3,x,w);
    check(fabs(x[1])<1e-15 and fabs(w[0]-1.0/3)<1e-15 and fabs(w[1]-4.0/3)<1e-15,"GLL nodes for n=3");
    gll_nodes_and_weights(5,x,w);
    check(fabs(x[1]+sqrt(3.0/7))<1e-15 and fabs(w[1]-49.0/90)<1e-15,"GLL nodes for n=5");
    for (auto shape:std::vector<std::array<uint32_t,3>>{{8,8,8},{1,8,8},{6,6,6},{10,10,10},{4,5,6},{1,1,7}})
    {
        test_shape<double>(shape,1e-12);
        test_shape<float>(shape,1e-4f);
    }
    printf(failures==0?"All tests passed\n":"%d tests failed\n",failures);
    return failures!=0;
}
