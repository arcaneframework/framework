// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephCuda.cc                                                (C) 2010-2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephArcane.h"
#include "arcane/aleph/cuda/IAlephCuda.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaAlephFactoryImpl
: public AbstractService
, public IAlephFactoryImpl
{
 public:
  CudaAlephFactoryImpl(const ServiceBuildInfo& sbi) : AbstractService(sbi){}
 public:
  virtual void initialize()
  {
    debug()<<"\t[AlephFactory::AlephFactory] cudaDeviceReset";
    cudaDeviceReset();
    debug()<<"\t[AlephFactory::AlephFactory] cudaDeviceSynchronize";
    cudaDeviceSynchronize();
    // Si le cublasInit ne pass pas, c'est que l'on a pas de device
    if (cublasInit()!=CUBLAS_STATUS_SUCCESS)
      return;
    //      throw FatalErrorException("AlephFactory", "Could not initialize CUBLAS!");
    // Now check if there is a device supporting CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
      throw FatalErrorException("AlephFactory", "No device found!");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      if (strncmp(deviceProp.name, "Device Emulation", 16))
        break;
    }
    if (dev == deviceCount)
      throw FatalErrorException(A_FUNCINFO, "No suitable device found");
    debug()<<"\t[Aleph::Cuda::device_check] setting device!";
    cudaSetDevice(dev);
  }

  virtual IAlephTopology* createTopology(ITraceMng* tm,AlephKernel* kernel, Integer index, Integer nb_row_size)
  {
    return 0;
  }

  virtual IAlephVector* createVector(ITraceMng* tm,AlephKernel* kernel, Integer index)
  {
    return new AlephVectorCnc(tm,kernel,index);
  }

  virtual IAlephMatrix* createMatrix(ITraceMng* tm,AlephKernel* kernel, Integer index)
  {
    return new AlephMatrixCnc(tm,kernel,index);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(CudaAlephFactoryImpl,IAlephFactoryImpl,CudaAlephFactory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
