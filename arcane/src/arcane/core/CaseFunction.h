// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunction.h                                              (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une fonction du jeu de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEFUNCTION_H
#define ARCANE_CORE_CASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

#include "arcane/utils/String.h"

#include "arcane/core/ICaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ISubDomain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour construire une instance de \a CaseFunction.
 * 
 * \param name nom de la fonction
 * \param param_type type du paramètre de la fonction
 * \param curve_type type de la courbe de la fonction
 */
class ARCANE_CORE_EXPORT CaseFunctionBuildInfo
{
 public:
  //! \deprecated Utiliser CaseFunctionBuildInfo(ITraceMng* tm,const String& name)
  ARCANE_DEPRECATED_260 CaseFunctionBuildInfo(ISubDomain* sd,const String& name);
  explicit CaseFunctionBuildInfo(ITraceMng* tm)
  : m_trace_mng(tm),
    m_param_type(ICaseFunction::ParamUnknown),
    m_value_type(ICaseFunction::ValueUnknown),
    m_deltat_coef(0.0)
  {}
  CaseFunctionBuildInfo(ITraceMng* tm,const String& name)
  : CaseFunctionBuildInfo(tm)
  {
    m_name = name;
  }

 public:

  ITraceMng* m_trace_mng; //!< Gestionnaire de trace associé.
  String m_name; //!< Nom de la fonction
  ICaseFunction::eParamType m_param_type; //!< Type du paramètre (x)
  ICaseFunction::eValueType m_value_type; //!< Type de la valeur (y)
  String m_transform_param_func; //!< Fonction de transformation X
  String m_transform_value_func; //!< Fonction de transformation Y
  Real m_deltat_coef; //!< Coefficient multiplicateur du deltat pour les tables en temps
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonction du jeu de données.
 *
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseFunction
: public ICaseFunction
, public ReferenceCounterImpl
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  //! Construit une fonction du jeu de données.
  explicit CaseFunction(const CaseFunctionBuildInfo& info);
  ~CaseFunction() override;

 public:

  String name() const override { return m_name; }
  void setName(const String& new_name) override;

  eParamType paramType() const  override{ return m_param_type; }
  void setParamType(eParamType type) override;

  eValueType valueType() const override { return m_value_type; }
  void setValueType(eValueType type) override;

  void setTransformValueFunction(const String& str) override;
  String transformValueFunction() const override { return m_transform_value_func; }

  void setTransformParamFunction(const String& str) override;
  String transformParamFunction() const override { return m_transform_param_func; }

  void setDeltatCoef(Real v) override { m_deltat_coef = v; }
  Real deltatCoef() const override { return m_deltat_coef; }

  bool checkIfValid() const override;

  Ref<ICaseFunction> toReference();
  ITraceMng* traceMng() const { return m_trace; }

 public:

 private:

  ITraceMng* m_trace; //!< Gestionnaire de traces
  String m_name; //!< Nom de la fonction
  eParamType m_param_type; //!< Type du paramètre (x)
  eValueType m_value_type; //!< Type de la valeur (y)
  String m_transform_param_func;  //!< Fonction de transformation du paramètre
  String m_transform_value_func;  //!< Fonction de transformation de la valeur
  Real m_deltat_coef;

 private:

 protected:
  
  template<typename ParamType> void _applyParamTransform2(ParamType& param) const;
  Real _applyValueComulTransform(Real v,Real comul) const;
  Integer _applyValueComulTransform(Integer v,Integer comul) const;
  String _applyValueComulTransform(const String& v,const String& comul) const;
  bool _applyValueComulTransform(bool v,bool comul) const;
  Real3 _applyValueComulTransform(Real3 v,Real3 comul) const;

  void _applyValueTransform(Real& value) const;
  void _applyValueTransform(Integer& value) const;
  void _applyValueTransform(String& value) const;
  void _applyValueTransform(Real3& value) const;
  void _applyValueTransform(bool& value) const;
  template<typename ValueType> void _applyValueTransform2(ValueType& value) const;
  void _applyParamTransform(Real& value) const;
  void _applyParamTransform(Integer& value) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
