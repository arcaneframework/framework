// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SUBDOMAINMODELPROPERTY_H
#define SUBDOMAINMODELPROPERTY_H

// #include "NumericalModel/FluxModel/FluxModel.h"
#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"
#include "Numerics/DiscreteOperator/IAdvectionOperator.h"
#include "NumericalModel/SubDomainModel/NumericalDomain/NumericalDomainImpl.h"

using namespace Arcane;

class FluxModel;
class IDivKGradDiscreteOperator ;
template<typename SchemeType> class FluxModelT;

struct SubDomainModelProperty
{
  //!Define type of NumericalDomain
  typedef NumericalDomainImpl NumericalDomain ;
  
  //!Define allowed boundary condition type
  typedef enum {
    Dirichlet, 
    Neumann,
    OverlapDirichlet,
    NullFlux,
    UnknownBC
  } eBoundaryConditionType ;
  
  typedef eBoundaryConditionType BCType ;
  
  //!Define Flux model type used for boundary condition
  typedef FluxModel FluxModelType ;
  typedef FluxModelT<IDivKGradDiscreteOperator> DiffFluxModelType;
  typedef FluxModelT<IAdvectionOperator> AdvFluxModelType;
  
  typedef enum {
    Flux,
    Gradient,
    UndefinedBCValueType
  } eBoundaryConditionValueType ;
  
  typedef eBoundaryConditionValueType BCValueType ;
  
  static const BCValueType defBCValueType()
  { return Flux ; }
  
  typedef enum {
    Pressure,
    Saturation,
    Composition,
    UConcentration,
    VConcentration
  } eRealVarType ;
};
#endif /*DOMAINMODELPROPERTY_H */
