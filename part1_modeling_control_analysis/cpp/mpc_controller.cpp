#include <Eigen/Dense>
#include <osqp/osqp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using RM = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template<class T>
void readn(std::ifstream& f,T* p,std::size_t n){
    f.read(reinterpret_cast<char*>(p), n*sizeof(T));
    if(!f) throw std::runtime_error("Unexpected end of binary file");
}
double p99(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double x=.99*(v.size()-1); size_t a=std::floor(x),b=std::ceil(x);
    return v[a]+(v[b]-v[a])*(x-a);
}
struct Data{
    int64_t Np,Nc,nx,nu,nd,ns; double dt,rho,ea,er;
    Eigen::VectorXd xeq,teq,lim;
    RM P,Az,bref,tff,xpy,tpy;
};
Data load(const std::string& path){
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open "+path);
    int64_t h[8]; double z[4]; readn(f,h,8); readn(f,z,4);
    if(h[0]!=0x555235454D504331LL || h[1]!=1) throw std::runtime_error("Bad binary header");
    Data d; d.Np=h[2];d.Nc=h[3];d.nx=h[4];d.nu=h[5];d.nd=h[6];d.ns=h[7];
    d.dt=z[0];d.rho=z[1];d.ea=z[2];d.er=z[3];
    d.xeq.resize(d.nx);d.teq.resize(d.nu);d.lim.resize(d.nu);
    d.P.resize(d.nd,d.nd);d.Az.resize(d.nd,d.nx);d.bref.resize(d.ns,d.nd);
    d.tff.resize(d.ns+d.Np,d.nu);d.xpy.resize(d.ns,d.nx);d.tpy.resize(d.ns,d.nu);
    readn(f,d.xeq.data(),d.xeq.size());readn(f,d.teq.data(),d.teq.size());readn(f,d.lim.data(),d.lim.size());
    readn(f,d.P.data(),d.P.size());readn(f,d.Az.data(),d.Az.size());readn(f,d.bref.data(),d.bref.size());
    readn(f,d.tff.data(),d.tff.size());readn(f,d.xpy.data(),d.xpy.size());readn(f,d.tpy.data(),d.tpy.size());
    return d;
}
int main(int argc,char**argv){
 try{
    auto d=load(argc>1?argv[1]:"results/mpc/cpp_data/mpc_cpp_data.bin");
    const OSQPInt n=d.nd,m=d.nd;

    std::vector<OSQPFloat> Px; std::vector<OSQPInt> Pi,Pp(n+1);
    for(OSQPInt j=0;j<n;j++){Pp[j]=Px.size();for(OSQPInt i=0;i<=j;i++){Px.push_back(d.P(i,j));Pi.push_back(i);}}
    Pp[n]=Px.size();
    OSQPCscMatrix* P=OSQPCscMatrix_new(n,n,Px.size(),Px.data(),Pi.data(),Pp.data());

    std::vector<OSQPFloat> Ax(n,1),q(n),lo(n),hi(n);
    std::vector<OSQPInt> Ai(n),Ap(n+1);
    for(OSQPInt j=0;j<n;j++){Ai[j]=j;Ap[j]=j;} Ap[n]=n;
    OSQPCscMatrix* A=OSQPCscMatrix_new(m,n,n,Ax.data(),Ai.data(),Ap.data());

    OSQPSettings st; osqp_set_default_settings(&st);
    st.verbose=0;st.polishing=0;st.warm_starting=1;st.eps_abs=d.ea;st.eps_rel=d.er;
    std::fill(lo.begin(),lo.end(),-OSQP_INFTY);std::fill(hi.begin(),hi.end(),OSQP_INFTY);
    OSQPSolver* s=nullptr;
    OSQPInt flag=osqp_setup(&s,P,q.data(),A,lo.data(),hi.data(),m,n,&st);
    if(flag) throw std::runtime_error(osqp_error_message(flag));

    double maxdiff=0; int64_t maxk=0,maxj=0; std::vector<double> tm; std::vector<double> its;
    tm.reserve(d.ns);its.reserve(d.ns);

    for(int64_t k=0;k<d.ns;k++){
      auto tic=std::chrono::steady_clock::now();
      Eigen::VectorXd g=d.Az*(d.xpy.row(k).transpose()-d.xeq)+d.bref.row(k).transpose();
      for(int64_t i=0;i<d.nd;i++) q[i]=g[i];

      for(int64_t j=0;j<d.Nc-1;j++) for(int64_t a=0;a<d.nu;a++){
        int64_t id=j*d.nu+a, r=k+j;
        lo[id]=-d.lim[a]-d.tff(r,a); hi[id]=d.lim[a]-d.tff(r,a);
      }
      for(int64_t a=0;a<d.nu;a++){
        double l=-OSQP_INFTY,u=OSQP_INFTY;
        for(int64_t h=d.Nc-1;h<d.Np;h++){
          l=std::max(l,-d.lim[a]-d.tff(k+h,a));
          u=std::min(u, d.lim[a]-d.tff(k+h,a));
        }
        lo[(d.Nc-1)*d.nu+a]=l;hi[(d.Nc-1)*d.nu+a]=u;
      }

      flag=osqp_update_data_vec(s,q.data(),lo.data(),hi.data()); if(flag) throw std::runtime_error("update failed");
      flag=osqp_solve(s); if(flag) throw std::runtime_error("solve failed");
      if(s->info->status_val!=OSQP_SOLVED && s->info->status_val!=OSQP_SOLVED_INACCURATE)
        throw std::runtime_error(std::string("OSQP: ")+s->info->status);

      for(int64_t a=0;a<d.nu;a++){
        double tau=d.tff(k,a)+s->solution->x[a], df=std::abs(tau-d.tpy(k,a));
        if(df>maxdiff){maxdiff=df;maxk=k;maxj=a;}
      }
      auto toc=std::chrono::steady_clock::now();
      tm.push_back(std::chrono::duration<double,std::milli>(toc-tic).count());
      its.push_back(s->info->iter);
    }
    double mean=std::accumulate(tm.begin(),tm.end(),0.0)/tm.size();
    double imean=std::accumulate(its.begin(),its.end(),0.0)/its.size();
    std::cout<<std::setprecision(12);
    std::cout<<"MPC C++ CROSS-VALIDATION\n";
    std::cout<<"OSQP version       : "<<osqp_version()<<"\n";
    std::cout<<"Samples            : "<<d.ns<<"\n";
    std::cout<<"Max |tau_cpp-tau_py| [Nm]: "<<maxdiff<<"  (k="<<maxk<<", joint="<<maxj+1<<")\n";
    std::cout<<"Iterations mean    : "<<imean<<"\n";
    std::cout<<"Timing mean [ms]   : "<<mean<<"\n";
    std::cout<<"Timing P99  [ms]   : "<<p99(tm)<<"\n";
    std::cout<<"Timing max  [ms]   : "<<*std::max_element(tm.begin(),tm.end())<<"\n";
    std::cout<<"2 ms utilization mean [%]: "<<100*mean/d.dt/1000.0<<"\n";
    std::cout<<"Validation         : "<<(maxdiff<1e-4?"PASS":"FAIL")<<"\n";
    osqp_cleanup(s);OSQPCscMatrix_free(P);OSQPCscMatrix_free(A);
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
