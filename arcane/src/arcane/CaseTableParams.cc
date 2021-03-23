// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTableParams.cc                                          (C) 2000-2016 */
/*                                                                           */
/* Paramètre d'une fonction du jeu de données.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/String.h"

#include "arcane/datatype/SmallVariant.h"

#include "arcane/CaseTableParams.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICFParamSetter
{
 public:
  typedef UniqueArray<SmallVariant> Params;
 public:
  ICFParamSetter(Params* v) : m_param_list(v) {}
  virtual ~ICFParamSetter() {}
 public:
  virtual void value(Integer id,Real& v) const =0;
  virtual void value(Integer id,Integer& v) const =0;
  virtual void value(Integer id,bool& v) const =0;
  virtual CaseTable::eError appendValue(const String& value) =0;
  virtual CaseTable::eError setValue(Integer id,const String& value) =0;
  virtual CaseTable::eError setValue(Integer id,Real v) =0;
  virtual CaseTable::eError setValue(Integer id,Integer v) =0;
  virtual CaseTable::eError setValue(Integer id,bool v) =0;
  virtual void removeValue(Integer id) =0;
  virtual void toString(Integer id,String& str) const =0;
 public:
 protected:
  
  const SmallVariant& param(Integer id) const
  { return (*m_param_list)[id]; }

  SmallVariant& param(Integer id)
  { return (*m_param_list)[id]; }

  Params& params()
  { return *m_param_list; }

  Integer nbElement() const
  { return m_param_list->size(); }

 private:

  Params* m_param_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type>
class CFParamSetterT
: public ICFParamSetter
, private VariantGetterT<Type>
{
 public:
  using VariantGetterT<Type>::asType;

  CFParamSetterT(Params* v)
  : ICFParamSetter(v) {}
  virtual ~CFParamSetterT() {}

 public:
  virtual void value(Integer id,Real& v) const;
  virtual void value(Integer id,Integer& v) const;
  virtual void value(Integer id,bool& v) const;
  virtual CaseTable::eError appendValue(const String& value);
  virtual CaseTable::eError setValue(Integer id,const String& value);
  virtual CaseTable::eError setValue(Integer id,Real v);
  virtual CaseTable::eError setValue(Integer id,Integer v);
  virtual CaseTable::eError setValue(Integer id,bool v);
  virtual void removeValue(Integer id);
  virtual void toString(Integer id,String& str) const;

 private:
  bool _checkConvert(Integer id,const String& str,Type& value) const;
  CaseTable::eError _checkValid(Integer id,Type value) const;
  CaseTable::eError _setIfValid(Integer id,Type value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void CFParamSetterT<Type>::
value(Integer id,Real& v) const 
{
  v = (Real)(asType(param(id)));
}
template<class Type> void CFParamSetterT<Type>::
value(Integer id,Integer& v) const
{
  v = Convert::toInteger(asType(param(id)));
}
template<class Type> void CFParamSetterT<Type>::
value(Integer id,bool& v) const
{
  v = Convert::toBool(asType(param(id)));
}
template<class Type> void CFParamSetterT<Type>::
toString(Integer id,String& s) const
{
  s = String::fromNumber(asType(param(id)));
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
setValue(Integer id,Real v)
{
  return _setIfValid(id,(Type)(Convert::toDouble(v)));
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
setValue(Integer id,Integer v)
{
  return _setIfValid(id,(Type)(v));
}
template<class Type> CaseTable::eError CFParamSetterT<Type>::
setValue(Integer id,bool v)
{
  return _setIfValid(id,(Type)(v));
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
setValue(Integer id,const String& s)
{
  Type avalue = Type();
  if (_checkConvert(id,s,avalue))
    return CaseTable::ErrCanNotConvertParamToRightType;
  return _setIfValid(id,avalue);
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
appendValue(const String& s)
{
  Type avalue = Type();
  Integer id = nbElement();
  if (_checkConvert(id,s,avalue))
    return CaseTable::ErrCanNotConvertParamToRightType;
  CaseTable::eError err = _checkValid(id,avalue);
  if (err==CaseTable::ErrNo){
    SmallVariant v; v.setValueAll(avalue);
    params().add(v);
  }
  return err;
}

template<class Type> void CFParamSetterT<Type>::
removeValue(Integer index)
{
  params().remove(index);
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
_setIfValid(Integer id,Type avalue)
{
  CaseTable::eError err = _checkValid(id,avalue);
  if (err==CaseTable::ErrNo)
    param(id).setValueAll(avalue);
  return err;
}

template<class Type> bool CFParamSetterT<Type>::
_checkConvert(Integer id,const String& str,Type& avalue) const
{
  ARCANE_UNUSED(id);
  avalue = Type();
  if (!str.null()){
    if (builtInGetValue(avalue,str))
      return true;
  }
  return false;
}

template<class Type> CaseTable::eError CFParamSetterT<Type>::
_checkValid(Integer id,Type avalue) const
{
  Integer nb_param = nbElement();
  // Vérifie que le 'begin' courant est supérieur au précédent.
  if (nb_param!=0 && id>0){
    Type previous_value = asType(param(id-1));
    if (avalue<previous_value)
      return CaseTable::ErrNotGreaterThanPrevious;
  }

  // Vérifie que le 'begin' courant est inférieur au suivant.
  if (nb_param!=0 && (id+1)<nb_param){
    Type next_value = asType(param(id+1));
    if (avalue>next_value)
      return CaseTable::ErrNotLesserThanNext;
  }

  return CaseTable::ErrNo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseTableParams::Impl
{
 public:
  friend class CaseTableParams;

  typedef UniqueArray<SmallVariant> Params;

 public:

  Impl(CaseTable::eParamType type);
  ~Impl();

 public:

  void setType(CaseTable::eParamType type);

 private:

  CaseTable::eParamType m_param_type;
  ICFParamSetter* m_setter;                  
  Params m_param_list; //!< Liste des valeurs
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTableParams::Impl::
Impl(CaseTable::eParamType type)
: m_param_type(CaseTable::ParamUnknown)
, m_setter(nullptr)
{
  setType(type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTableParams::Impl::
~Impl()
{
  delete m_setter;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseTableParams::Impl::
setType(CaseTable::eParamType type)
{
  //cerr << "** TRY TO TYPE CHANGED " << (int)type << ' '  << (int)m_param_type << "\n";
  if (type!=m_param_type)
    delete m_setter;

  /*if (m_param_type==CaseTable::TReal && type==CaseTable::TInteger){
    for( Integer i=0, s=m_param_list.size(); i<s; ++i ){
      Integer v = static_cast<Integer>(m_param_list[i].asReal());
      m_param_list[i].setValue(v);
    }
    }*/
  
  m_param_type = type;
  //cerr << "** TYPE CHANGED\n";
  switch(m_param_type){
  case CaseTable::ParamUnknown:
  case CaseTable::ParamReal:
    m_setter = new CFParamSetterT<Real>(&m_param_list);
    break;
  case CaseTable::ParamInteger:
    m_setter = new CFParamSetterT<Integer>(&m_param_list);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseTableParams::
CaseTableParams(CaseTable::eParamType v)
: m_p(new Impl(v))
{
}

CaseTableParams::
~CaseTableParams()
{
  delete m_p;
}

bool CaseTableParams::
null() const
{
  return m_p->m_param_type==CaseTable::ParamUnknown;
}

Integer CaseTableParams::
nbElement() const
{
  return m_p->m_param_list.size();
}

void CaseTableParams::
value(Integer id,Real& v) const
{
  m_p->m_setter->value(id,v);
}

void CaseTableParams::
value(Integer id,Integer& v) const
{
  m_p->m_setter->value(id,v);
}

void CaseTableParams::
value(Integer id,bool& v) const
{
  m_p->m_setter->value(id,v);
}

CaseTable::eError CaseTableParams::
appendValue(const String& avalue)
{
  return m_p->m_setter->appendValue(avalue);
}

CaseTable::eError CaseTableParams::
setValue(Integer id,const String& avalue)
{
  return m_p->m_setter->setValue(id,avalue);
}

CaseTable::eError CaseTableParams::
setValue(Integer id,Real v)
{
  return m_p->m_setter->setValue(id,v);
}

CaseTable::eError CaseTableParams::
setValue(Integer id,Integer v)
{
  return m_p->m_setter->setValue(id,v);
}

CaseTable::eError CaseTableParams::
setValue(Integer id,bool v)
{
  return m_p->m_setter->setValue(id,v);
}

void CaseTableParams::
removeValue(Integer id)
{
  m_p->m_setter->removeValue(id);
}

void CaseTableParams::
toString(Integer id,String& str) const
{
  m_p->m_setter->toString(id,str);
}

void CaseTableParams::
setType(CaseTable::eParamType new_type)
{
  m_p->setType(new_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class CFParamSetterT<Real>;
template class CFParamSetterT<Integer>;
template class CFParamSetterT<bool>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

