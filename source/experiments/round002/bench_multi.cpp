// Laboratory timing only. Load all candidates once; alternate queued executions on one logical core.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { auto bench_status=(x); if(bench_status!=NRT_SUCCESS) {fprintf(stderr,"%s: %d\n",#x,int(bench_status));exit(1);} }while(0)
struct Model { std::string name; nrt_model_t* m; nrt_tensor_set_t *in,*out; };
static double run(Model& m, int count) {
  std::vector<NRT_STATUS> rets(count); nrta_seq_t seq=0;
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;i++) {
    NRT_STATUS s;
    while((s=nrta_execute_schedule(m.m,m.in,m.out,0,&rets[i],&seq))==NRT_QUEUE_FULL) {}
    CK(s);
  }
  bool done=false; while(!done) CK(nrta_is_completed(seq,&done));
  auto end=std::chrono::steady_clock::now();
  for(auto s:rets) CK(s);
  return std::chrono::duration<double,std::micro>(end-start).count()/count;
}
int main(int argc,char** argv) {
  if(argc<5 || argc%2!=1) return 2; // rounds count name neff [name neff ...]
  int rounds=atoi(argv[1]), count=atoi(argv[2]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""));
  std::vector<Model> models;
  for(int a=3;a<argc;a+=2) {
    Model m; m.name=argv[a];
    std::ifstream f(argv[a+1],std::ios::binary);
    if(!f) return 2;
    std::vector<char> neff((std::istreambuf_iterator<char>(f)),{});
    CK(nrt_load(neff.data(),neff.size(),0,1,&m.m));
    CK(nrt_allocate_tensor_set(&m.in)); CK(nrt_allocate_tensor_set(&m.out));
    nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m.m,&info));
    for(uint64_t i=0;i<info->tensor_count;i++) {
      auto& ti=info->tensor_array[i]; nrt_tensor_t* t;
      CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE,0,ti.size,ti.name,&t));
      std::vector<char> zeros(ti.size,0); CK(nrt_tensor_write(t,zeros.data(),0,ti.size));
      CK(nrt_add_tensor_to_tensor_set(ti.usage==NRT_TENSOR_USAGE_INPUT?m.in:m.out,ti.name,t));
    }
    models.push_back(m);
  }
  // All these lab kernels are stable on zero inputs: no position increment or weight mutation.
  for(auto& m:models) run(m,10);
  for(int r=0;r<rounds;r++) {
    for(size_t j=0;j<models.size();j++) {
      auto& m=models[r%2?models.size()-1-j:j];
      printf("{\"round\":%d,\"name\":\"%s\",\"us\":[",r,m.name.c_str());
      for(int rep=0;rep<3;rep++) printf("%s%.3f",rep?",":"",run(m,count));
      printf("]}\n"); fflush(stdout);
    }
  }
  nrt_close();
}
