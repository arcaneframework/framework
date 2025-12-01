#include "arccore/base/ArrayView.h"
#include "arccore/base/String.h"

#include "arccore/common/accelerator/IReduceMemoryImpl.h"
#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/CommonUtils.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arccore/accelerator/Reduce.h"

#include <limits.h>
#include <float.h>
#include <atomic>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/GenericReducer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
