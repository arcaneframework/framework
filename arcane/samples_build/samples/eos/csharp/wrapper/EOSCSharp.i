%module(directors="1") EOSCharp

%import core/ArcaneSwigCore.i

%typemap(csimports) SWIGTYPE
%{
using Arcane;
%}

%{
#include "ArcaneSwigUtils.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "IEquationOfState.h"
using namespace Arcane;
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DECLARE_INTERFACE(EOS,IEquationOfState)

%include IEquationOfState.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_SWIG_DEFINE_SERVICE(EOS,IEquationOfState,
                           public abstract void InitEOS(CellGroup group,
                                                        VariableCellReal pressure,
                                                        VariableCellReal adiabatic_cst,
                                                        VariableCellReal density,
                                                        VariableCellReal internal_energy,
                                                        VariableCellReal sound_speed);
                           public abstract void ApplyEOS(CellGroup group,
                                                         VariableCellReal adiabatic_cst,
                                                         VariableCellReal density,
                                                         VariableCellReal internal_energy,
                                                         VariableCellReal pressure,
                                                         VariableCellReal sound_speed);
                           );

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
