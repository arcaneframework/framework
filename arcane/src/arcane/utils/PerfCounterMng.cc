// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PerfCounterMng.cc                                           (C) 2000-2017 */
/*                                                                           */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/PerfCounterMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retourne la fréquence du CPU en Mhz.
 *
 * Ne marche que sous Linux et ne tient pas compte des variations
 * possibles de fréquence telles que le turbo-boost.
 */
extern "C++" int
arcaneGetCpuBaseFrequency()
{
#ifndef WIN32
  /* return cpu frequency in MHZ as read in /proc/cpuinfo */
  float ffreq = 0;
  int r = 0;
  char *rr = NULL;
  FILE *fdes = fopen("/proc/cpuinfo","r");
  char buff[256];
  int bufflength = 256;
  do{
    rr = fgets(buff,bufflength,fdes);
    r = sscanf(buff,"cpu MHz         : %f\n",&ffreq);
    if(r==1){
      break;
    }
  } while(rr != NULL);

  fclose(fdes);

  int ifreq = (int)ffreq;
  return ifreq;
#else
  std::cerr << "getCpuFreq not functionnal under win\n";
  return 1;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

