// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumericWrapper.h                                            (C) 2000-2025 */
/*                                                                           */
/* Wrapper pour swig.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_WRAPPER_CORE_NUMERICWRAPPER_H
#define ARCANE_WRAPPER_CORE_NUMERICWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Ce wrapper ne doit être utilisé que par le code généré par swig
// Les types *_POD doivent pouvoir être utilisés depuis le C (C-linkage)
// et donc utiliser les constructeurs par défaut (pour le constructeur vide et par recopie).
// Avec les versions de swig testées (jusqu'à 3.0.12), le code suivant est généré dans
// certaines méthodes utilisant les directors:
//   Arcane::Real2_POD v = 0;
// Pour que cela compile, on ajoute donc un constructeur qui prend un entier en argument.
// Comme cela ne sert que pour la déclaration de la variable, cela ne pose pas de problèmes.

#include "arcane/utils/NumericTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class Real2_POD
{
 public:
  // Pour SWIG
  Real2_POD(int){}
  Real2_POD() = default;
  operator Real2()
  {
    return Real2(x,y);
  }
 public:
  Real x;
  Real y;
};

class Real3_POD
{
 public:
  // Pour SWIG
  Real3_POD(int){}
  Real3_POD() = default;
  operator Real3()
  {
    return Real3(x,y,z);
  }
 public:
  Real x;
  Real y;
  Real z;
};

class Real2x2_POD
{
 public:
  Real2x2_POD(int){}
  Real2x2_POD() = default;
  operator Real2x2()
  {
    return Real2x2(x,y);
  }
 public:
  Real2_POD x;
  Real2_POD y;
};

class Real3x3_POD
{
 public:
  Real3x3_POD(int){}
  Real3x3_POD() = default;
  operator Real3x3()
  {
    return Real3x3(x,y,z);
  }
 public:
  Real3_POD x;
  Real3_POD y;
  Real3_POD z;
};

static inline Real2 _PODToReal(const Real2_POD& pod)
{
  return Real2(pod.x,pod.y);
}
static inline Real3 _PODToReal(const Real3_POD& pod)
{
  return Real3(pod.x,pod.y,pod.z);
}
static inline Real3x3 _PODToReal(const Real3x3_POD& pod)
{
  return Real3x3(_PODToReal(pod.x),_PODToReal(pod.y),_PODToReal(pod.z));
}

static inline Real2x2 _PODToReal(const Real2x2_POD& pod)
{
  return Real2x2(_PODToReal(pod.x),_PODToReal(pod.y));
}
static inline Real2_POD _RealToPOD(const Real2& r)
{
  Real2_POD pod;
  pod.x = r.x;
  pod.y = r.y;
  return pod;
}
static inline Real3_POD _RealToPOD(const Real3& r)
{
  Real3_POD pod;
  pod.x = r.x;
  pod.y = r.y;
  pod.z = r.z;
  return pod;
}

static inline Real3x3_POD _RealToPOD(const Real3x3& r)
{
  Real3x3_POD pod;
  pod.x = _RealToPOD(r.x);
  pod.y = _RealToPOD(r.y);
  pod.z = _RealToPOD(r.z);
  return pod;
}

static inline Real2x2_POD _RealToPOD(const Real2x2& r)
{
  Real2x2_POD pod;
  pod.x = _RealToPOD(r.x);
  pod.y = _RealToPOD(r.y);
  return pod;
}

}

#include "arcane/ItemTypes.h"
#include "arcane/ItemInternal.h"

namespace Arcane
{
  class CaseOptionBase;

#ifndef SWIG
  // Cette classe sert de type de retour pour wrapper la class ArrayView
  class ArrayViewPOD
  {
    public:
     Integer m_size;
     void* m_ptr;
  };

  // Cette classe sert de type d'entrée retour pour wrapper la class ArrayView
  template<typename DataType> class ArrayViewPOD_T
  {
   public:
    // Pour SWIG, ajout du constructeur avec un entier.
    ArrayViewPOD_T(int){}
    ArrayViewPOD_T() = default;
   public:
    Integer m_size;
    DataType* m_ptr;
  };

  // Cette classe sert de type de retour pour wrapper la class ConstArrayView
  class ConstArrayViewPOD
  {
    public:
     Integer m_size;
     const void* m_ptr;
  };

  // Cette classe sert de type de retour pour wrapper la class ConstArrayView
  template<typename DataType> class ConstArrayViewPOD_T
  {
   public:
    // Pour SWIG, ajout du constructeur avec un entier.
    ConstArrayViewPOD_T(int){}
    ConstArrayViewPOD_T() = default;
   public:
     Integer m_size;
     const DataType* m_ptr;
  };

// Cette classe sert de type de retour pour wrapper la class Array2View
  class Array2ViewPOD
  {
   public:
    void* m_ptr;
    Integer m_dim1_size;
    Integer m_dim2_size;
  };
#endif

  // Cette classe sert de type de retour pour wrapper la classe 'ItemEnumerator'
  class ItemEnumeratorPOD
  {
   public:
    ItemSharedInfo* m_shared_info;
    const Int32* m_local_ids;
    Integer m_index;
    Integer m_count;
  };

  // Cette classe sert de type de retour pour wrapper la classe 'ItemIndexArrayView'
  class ItemIndexArrayViewPOD
  {
   public:
    ConstArrayViewPOD_T<Int32> m_local_ids;
    Int32 m_flags;
  };

  // Cette classe sert de type de retour pour wrapper la classe 'ItemVectorView'
  // Note: cette structure doit avoir le même layout que la
  // version qui est dans NumericWrapper.h
  class ItemVectorViewPOD
  {
   public:
    ItemIndexArrayViewPOD m_local_ids;
    ItemSharedInfo* m_shared_info;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  extern ARCANE_CORE_EXPORT bool
  _caseOptionConvert(const CaseOptionBase& co,const String& name,ItemGroup& obj);
  extern ARCANE_CORE_EXPORT bool
  _caseOptionConvert(const CaseOptionBase& co,const String& name,NodeGroup& obj);
  extern ARCANE_CORE_EXPORT bool
  _caseOptionConvert(const CaseOptionBase& co,const String& name,EdgeGroup& obj);
  extern ARCANE_CORE_EXPORT bool
  _caseOptionConvert(const CaseOptionBase& co,const String& name,FaceGroup& obj);
  extern ARCANE_CORE_EXPORT bool
  _caseOptionConvert(const CaseOptionBase& co,const String& name,CellGroup& obj);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
