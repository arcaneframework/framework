// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SDMBOUNDARYCONDITION_H_
#define SDMBOUNDARYCONDITION_H_

#include "NumericalModel/Utils/OpT.h"
#include "NumericalModel/SubDomainModel/IDiscreteVarTypes.h"

#include "Numerics/DiscreteOperator/IDivKGradDiscreteOperator.h"
#include "Numerics/DiscreteOperator/IAdvectionOperator.h"

#include <arcane/ItemGroupRangeIterator.h>
#include <arcane/ArcaneVersion.h>

using namespace Arcane;

class IDivKGradDiscreteOperator ;
template<typename SchemeType> class FluxModelT;

//#define DEBUG_INFO

template<IDiscreteVarTypes::eVarDim dim>
class BCValues
{
public :
   template<typename BoundaryCondition>
static  typename IDiscreteVarTypes::template ValueArrayType<dim>::type&
   values(BoundaryCondition* bc) ;
   template<typename BoundaryCondition>
   static
   typename IDiscreteVarTypes::template ValueArrayType<dim>::type const&
   values(const BoundaryCondition* bc) ;
};


template<>
class BCValues<IDiscreteVarTypes::Scalar>
{
public :
  template<typename BoundaryCondition>
  static IDiscreteVarTypes::template ValueArrayType<IDiscreteVarTypes::Scalar>::type&
  values(BoundaryCondition* bc)
  {
    return bc->scalarValues() ;
  }
  template<typename BoundaryCondition>
  static IDiscreteVarTypes::template ValueArrayType<IDiscreteVarTypes::Scalar>::type const&
  values(const BoundaryCondition* bc)
  {
    return bc->scalarValues() ;
  }
} ;

template<>
class BCValues<IDiscreteVarTypes::Vectorial>
{
public :
  template<typename BoundaryCondition>
  static
  IDiscreteVarTypes::template ValueArrayType<IDiscreteVarTypes::Vectorial>::type&
  values(BoundaryCondition* bc)
  {
    return bc->vectorialValues() ;
  }
  template<typename BoundaryCondition>
  static
  IDiscreteVarTypes::template ValueArrayType<IDiscreteVarTypes::Vectorial>::type const&
  values(const BoundaryCondition* bc)
  {
    return bc->vectorialValues() ;
  }
} ;


template<class Item,class ModelProperty>
class ModelBoundaryCondition
{
public :
  typedef typename ModelProperty::eBoundaryConditionType Type ;
  typedef ItemGroupT<Item> BoundaryType ;
  typedef ModelBoundaryCondition<Item,ModelProperty>  BoundaryConditionType ;
  typedef IDiscreteVarTypes::eVarDim VarDim ;
  typedef FluxModelT<IDivKGradDiscreteOperator> DiffFluxModelType;
  typedef FluxModelT<IAdvectionOperator> AdvFluxModelType;
  ModelBoundaryCondition(Integer id,
                         Type bc_type,
                         Integer boundary_id,
                         IDiscreteVarTypes::eVarDim bc_value_dim=IDiscreteVarTypes::Scalar)
  : m_id(id)
  , m_type(bc_type)
  , m_boundary_id(boundary_id)
  , m_flux_model(NULL)
  , m_adv_flux_model(NULL)
  , m_diff_flux_model(NULL)
  , m_is_active(false)
  , m_is_uniform(true)
  , m_value_type(ModelProperty::defBCValueType())
  , m_value_dim(bc_value_dim)
  , m_value(0.)
  {
  }
  virtual ~ModelBoundaryCondition() {}

  Integer getId() { return m_id ; }

  Type getType() { return m_type ; }

  Integer getBoundaryId() { return m_boundary_id ;}

  BoundaryType& getBoundary() { return m_boundary ; }

  void setBoundary(BoundaryType boundary)
  {
    m_boundary = boundary ;
  }

  typename ModelProperty::FluxModelType* getFluxModel()
  {
    return m_flux_model ;
  }

  
  typename ModelProperty::AdvFluxModelType* getAdvFluxModel()
    {
      return m_adv_flux_model;
    }

  typename ModelProperty::DiffFluxModelType* getDiffFluxModel()
    {
      return m_diff_flux_model;
    }

  void setFluxModel(typename ModelProperty::FluxModelType* flux_model)
  {
    m_flux_model = flux_model ;
  }

  void setAdvFluxModel(typename ModelProperty::AdvFluxModelType* adv_flux_model)
    {
      m_adv_flux_model = adv_flux_model;
    }

  void setDiffFluxModel(typename ModelProperty::DiffFluxModelType* diff_flux_model)
    {
      m_diff_flux_model = diff_flux_model;
    }

  bool isActive()
  { return m_is_active ; }

  void activate(bool flag)
  {
    m_is_active = flag ;
  }


  template<typename BoundaryValues,IDiscreteVarTypes::eVarDim dim>
  void copyValues(BoundaryValues& values)
  {
    typedef typename IDiscreteVarTypes::template ValueArrayType<dim>::type VarArrayType ;
    IDiscreteVarTypes::AssignOp<dim> assign ;
    VarArrayType& bc_values = BCValues<dim>::values(this) ;
    Integer k = 0 ;
    for( ItemGroupRangeIteratorT<Item> i(m_boundary); i.hasNext(); ++i )
    {
       assign(values[*i],bc_values[k++]) ;
    }
  }
  void setValue(Real value,
                typename ModelProperty::BCValueType value_type=ModelProperty::defBCValueType())
  {
    m_value_type = value_type ;
    m_value = value ;
    m_values.resize(0) ;
    m_vectorial_values.resize(0) ;
    m_is_uniform = true ;
  }


  void setValue(ConstArrayView<Real> values,
                typename ModelProperty::BCValueType value_type=ModelProperty::defBCValueType())
  {
    m_value_type = value_type ;
    m_value = 0 ;
    m_values.resize(values.size()) ;
    m_values.copy(values) ;
    m_vectorial_values.resize(0) ;
    m_is_uniform = true ;
  }
  void setValues(ConstArrayView<Real> values,
                 Integer size2,
                 typename ModelProperty::BCValueType value_type=ModelProperty::defBCValueType())
  {
    m_is_uniform = false ;
    m_value_type = value_type ;
    m_value_dim = IDiscreteVarTypes::Vectorial ;
    //IDiscreteVarTypes::AssignOp<IDiscreteVarTypes::Vectorial> assign ;
    m_vectorial_values.resize(m_boundary.size(),size2) ;
    cout<<"allocate "<<m_boundary.size()<<" "<<size2<<endl;
    for(Integer i=0;i<size2;++i)
      cout<<values[i]<<" ";
    cout<<endl;
    for(Integer k = 0 ;k<m_boundary.size();++k)
      m_vectorial_values[k].copy(values) ;
    m_value = 0 ;
  }
  template<typename BoundaryValues,IDiscreteVarTypes::eVarDim dim>
  void setValues(BoundaryValues& values,
                 typename ModelProperty::BCValueType value_type=ModelProperty::defBCValueType())
  {
    m_is_uniform = false ;
    m_value_type = value_type ;
    m_value_dim = dim ;
    typedef typename IDiscreteVarTypes::template ValueArrayType<dim>::type ValuesType ;
    IDiscreteVarTypes::AssignOp<dim> assign ;
    ValuesType& bc_values = BCValues<dim>::values(this) ;
    bc_values.resize(m_boundary.size()) ;
    Integer k = 0 ;
    for( ItemGroupRangeIteratorT<Item> i(m_boundary); i.hasNext(); ++i )
    {
      //bc_values[k++] = values[*i] ;
      assign(bc_values[k++],values[*i]) ;
    }
    m_value = 0 ;
  }

  template<typename BoundaryValues>
  void initValues(BoundaryValues& values)
  {
    if(m_is_uniform)
      for( ItemGroupRangeIteratorT<Item> i(m_boundary); i.hasNext(); ++i )
      {
        values[*i] = m_value ;
      }
    else
      copyValues<BoundaryValues,IDiscreteVarTypes::Scalar>(values) ;
  }

  template<typename BoundaryValues,IDiscreteVarTypes::eVarDim dim>
  void initValues(BoundaryValues& values)
  {
    if(m_is_uniform)
    {
      IDiscreteVarTypes::AssignOp<dim> assign ;
      for( ItemGroupRangeIteratorT<Item> i(m_boundary); i.hasNext(); ++i )
      {
        assign(values[*i],m_value) ;
      }
    }
    else
      copyValues<BoundaryValues,dim>(values) ;
  }


  class IBCOp : public IOpT<BoundaryConditionType>
  {
  public :
    virtual ~IBCOp() {}
    virtual bool status() = 0 ;
  };

  /*template<class Model>
  struct FuncType
  {
    typedef void(Model::*func_type)(BoundaryConditionType* bc) ;
  };*/
  template<class Model>
  class BCOp
   : public IBCOp
   , public OpTT<Model,BoundaryConditionType>
  {
  public :
    typedef typename ModelTraits<Model>::template FuncType<BoundaryConditionType>::funcT_type func_type ;
    BCOp(Model* model, func_type func,BoundaryConditionType* bc=NULL)
    : OpTT<Model,BoundaryConditionType>(model,func)
    , m_bc(bc)
    , m_status(false)
    {}
    virtual ~BCOp() {}
    bool status() { return m_status ; }
    void setStatus(bool status)
    { m_status = status ; }
    void compute(BoundaryConditionType* bc)
    {
       OpTT<Model,BoundaryConditionType>::compute(bc) ;
    }
    void compute()
    {
      if(m_bc)
        OpTT<Model,BoundaryConditionType>::compute(m_bc) ;
    }
    BoundaryConditionType* m_bc ;
  private :
    bool m_status ;

  };

  inline Array<Real>& scalarValues() {
    return m_values ;
  }
  inline const Array<Real>& scalarValues() const {
    return m_values ;
  }
  inline RealArray2& vectorialValues() {
    return m_vectorial_values ;
  }
  inline const RealArray2& vectorialValues() const {
    return m_vectorial_values ;
  }
private :
  //! condition id for the manager
  Integer m_id ;

  //!boundary condition type
  Type m_type ;

  //! boundary id
  Integer m_boundary_id ;

  //!boundary item group
  BoundaryType m_boundary ;

  typename ModelProperty::FluxModelType* m_flux_model ;
  typename ModelProperty::AdvFluxModelType* m_adv_flux_model;
  typename ModelProperty::DiffFluxModelType* m_diff_flux_model;

  bool m_is_active ;
  bool m_is_uniform ;

  //!condition value description
  //!@{
  typename ModelProperty::BCValueType m_value_type ;
  IDiscreteVarTypes::eVarDim m_value_dim ;
  Real m_value ;
  Array<Real> m_values ;
  RealArray2 m_vectorial_values ;
  //!@}
};
#endif /*ModelBOUNDARYCONDITION_H_*/
