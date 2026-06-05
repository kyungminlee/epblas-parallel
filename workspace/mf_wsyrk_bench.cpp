/* Hot-path (SIMD, small K) timing for wsyrk: N=256, K=128. */
#include <multifloats.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
namespace mf = multifloats;
using T = mf::complex64x2;
static T cx(double r,double i){return T{{r,0.0},{i,0.0}};}
extern "C" void wsyrk_(const char*,const char*,const int*,const int*,const T*,
  const T*,const int*,const T*,T*,const int*,std::size_t,std::size_t);
int main(){
  const int N=256,K=128; const char L='L',NN='N';
  std::vector<T> A((size_t)N*K), C((size_t)N*N);
  for(int j=0;j<K;++j)for(int i=0;i<N;++i)A[(size_t)j*N+i]=cx(std::sin(0.3*i+0.7*j),std::cos(0.2*i+0.5*j));
  for(int j=0;j<N;++j)for(int i=0;i<N;++i)C[(size_t)j*N+i]=cx(0.1*i,0.1*j);
  const T al=cx(1.3,-0.2), be=cx(0.7,0.0);
  for(int w=0;w<3;++w) wsyrk_(&L,&NN,&N,&K,&al,A.data(),&N,&be,C.data(),&N,1,1);
  const int reps=200;
  auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<reps;++r) wsyrk_(&L,&NN,&N,&K,&al,A.data(),&N,&be,C.data(),&N,1,1);
  auto t1=std::chrono::steady_clock::now();
  double ns=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)reps;
  std::printf("wsyrk N=%d K=%d  %.0f ns/call  (checksum %.6f)\n",N,K,ns,(double)C[0].re.limbs[0]);
  return 0;
}
