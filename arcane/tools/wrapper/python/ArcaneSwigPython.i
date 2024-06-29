%module(directors="1") ArcanePython

%import core/ArcaneSwigCore.i

%{
#include "ArcaneSwigUtils.h"
#include "arcane/core/internal/VariableUtilsInternal.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/utils/MemoryView.h"
using namespace Arcane;
%}

%include arcane/core/internal/VariableUtilsInternal.h
%include arcane/core/internal/IDataInternal.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
