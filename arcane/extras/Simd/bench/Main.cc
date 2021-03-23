/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "bench/HydroBenchBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Arcane::HydroBenchBase*
createHydroBench();

extern "C++" Arcane::HydroBenchBase*
createNoVecHydroBench();

extern "C++" double
_getRealTime()
{
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME,&tp);
  double s = (double)tp.tv_sec;
  double ns = (double)tp.tv_nsec;
  return s + (ns / 1.0e9);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 *
 * This benchmark create a cartesian 3D mesh and do some operations on the
 * cells of the mesh.
 *
 * The benchmark uses specific classes to help the compiler vectorize the code.
 * These classes are named 'SimdSSE', 'SimdAVX' and 'SimdMIC' and can be
 * found in arcane/utils directory.
 *
 * There is two differents compute methods.
 * - the first one is called computeGeometricValues and is compute bound. It
 * computes several geometric values.
 * - the second one is called computeEquationOfState. The byte/flop ratio
 * is not very good and this part is more memory bound.
 *
 * The results of each part is diplayed at the end of run. The first part
 * is displayed 
 */
int
main()
{
  using namespace Arcane;
  
  HydroBenchBase* bench = createHydroBench();
  HydroBenchBase* novec_bench = createNoVecHydroBench();

  Int32 nx = 200;
  Int32 ny = 15;
  Int32 nz = 15;
  bench->allocate(nx,ny,nz);
  novec_bench->allocate(nx,ny,nz);

  int nb_loop = 20;
  int nb_eos_mul = 50; // Do more EOS compute because it is faster than computeGeometric

  novec_bench->computeGeometric(1);

  double t1 = _getRealTime();
  for( int i=0; i<nb_loop; ++i )
    bench->computeGeometric(1);
  double t2 = _getRealTime();
  for( int i=0; i<nb_loop; ++i )
    bench->computeGeometric(10);
  double t3 = _getRealTime();

  double diff_time1 = (t2-t1);
  double diff_time2 = (t3-t2);
  // Time for 'nb_loop' compute of cqs
  double cqs_time = (diff_time2-diff_time1) / 9.0;
  double prepare_time = (diff_time1 - cqs_time);

  bench->initForEquationOfState();
  novec_bench->initForEquationOfState();

  novec_bench->compare(bench);

  bench->computeEquationOfState(1);
  novec_bench->computeEquationOfState(1);

  // Compare with sequential version for correctness
  novec_bench->compare(bench);

  double eos_time = 0.0;
  {
    double t4 = _getRealTime();
    bench->computeEquationOfState(nb_loop*nb_eos_mul);
    double t5 = _getRealTime();
    eos_time = (t5-t4);
  }



  Int32 nb_cell = bench->nbCell();
  double mul = 1e9/ (nb_loop * nb_cell);
  const char* simd_name = bench->getSimdName();
  printf("SimdKind: %5s Cell: %10d - Clocks0: %lf s ClockCQS: %lf s ClockEOS: %lf s Prepare: %lf ns -- CQS: %lf ns EOS: %lf ns\n",
         simd_name,nb_cell,diff_time1, diff_time2,eos_time,prepare_time* mul,cqs_time * mul, eos_time * (mul / nb_eos_mul));
  return 0;
}
