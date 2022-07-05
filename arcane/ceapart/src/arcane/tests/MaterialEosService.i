%module(directors="1") EOSCharp

%import core/ArcaneSwigCore.i
%import materials/ArcaneSwigCeaMaterials.i

%typemap(csimports) SWIGTYPE
%{
using Arcane;
using Arcane.Materials;
%}

%{
#include "ArcaneSwigUtils.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/tests/IMaterialEquationOfState.h"
using namespace Arcane;
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DECLARE_INTERFACE(MaterialEos,IMaterialEquationOfState)

%include IMaterialEquationOfState.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_SWIG_DEFINE_SERVICE(MaterialEos,IMaterialEquationOfState,
                           public abstract void InitEOS(IMeshMaterial mat,
                                                        MaterialVariableCellReal pressure,
                                                        MaterialVariableCellReal density,
                                                        MaterialVariableCellReal internal_energy,
                                                        MaterialVariableCellReal sound_speed);
                           public abstract void ApplyEOS(IMeshMaterial mat,
                                                         MaterialVariableCellReal density,
                                                         MaterialVariableCellReal internal_energy,
                                                         MaterialVariableCellReal pressure,
                                                         MaterialVariableCellReal sound_speed);
                           );

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
