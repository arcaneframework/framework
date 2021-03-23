// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef AGS_UTILS_UTILS_H
#define AGS_UTILS_UTILS_H

#ifdef _MPI
#define MPICH_SKIP_MPICXX 1
//#include "mpi.h"
#endif

#include <arcane/ArcaneVersion.h>
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Limits.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Buffer.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/IIOMng.h"
#include "arcane/utils/Math.h"

#include <arcane/utils/FatalErrorException.h>

#include <arcane/utils/ITraceMng.h>
#include <arcane/IParallelMng.h>
#include <arcane/utils/Iostream.h>
#include <arcane/utils/Array.h>
#include <arcane/utils/Array2.h>

using namespace Arcane ;

#include <map>
#include <set>

// #if defined(arch_Linux) || defined(arch_SunOS) || defined(arch_IRIX)
// /* Ajoute un caractere '_' a la fin du nom de la fonction C */
// #define F2C(functionC) functionC##_
// #elif defined(arch_AIX)|| defined(WIN32)
// #define F2C(functionC) functionC
// #else
// #error "F2C macro needs to be defined for this architecture."
// /* #define F2C(functionC) functionC */
// #endif

#define FORMAT(w,p) std::setiosflags(ios::fixed)<<std::setw(w)<<std::setprecision(p)
#define FORMATS(w,p) std::setiosflags(ios::scientific)<<std::setw(w)<<std::setprecision(p)
#define FORMATW(w) std::setw(w)
#define FORMATF(n,c) std::setfill(c)<<std::setw(n)<<(c)<<std::setfill(' ')

#include "DefineUtils.h"

#include "ArcGeoSim.h"

#include "IVisitor.h"
#include "IOBuffer.h"
#include "Table.h"
#include "FunctionUtils.h"
#include "ArrayUtils.h"
// #include "ItemVector.h"

#endif /* AGS_UTILS_UTILS_H */
