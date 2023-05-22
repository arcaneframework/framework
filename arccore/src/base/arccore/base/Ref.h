// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Ref.h                                                       (C) 2000-2020 */
/*                                                                           */
/* Gestion des références sur une instance.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REF_H
#define ARCCORE_BASE_REF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExternalRef.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une référence à un service.
 */
class ARCCORE_BASE_EXPORT RefBase
{
 protected:
  class ARCCORE_BASE_EXPORT DeleterBase
  {
   protected:
    bool _destroyHandle(const void* instance,Internal::ExternalRef& handle);
    bool _destroyHandle(void* instance,Internal::ExternalRef& handle);
   private:
    bool _destroyHandleTrue(const void* instance,Internal::ExternalRef& handle);
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Internal
{
template<typename InstanceType>
class ReferenceCounterWrapper
{
  //! Vérifie que la classe 'InstanceType' a bien un typedef sur ReferenceCounterTag (\sa Ref)
  inline static void _checkHasReferenceCounterTag()
  {
    static_assert(std::is_same_v<typename InstanceType::ReferenceCounterTagType,ReferenceCounterTag>,"Bad tag");
  }
 public:
  template<typename U>
  ReferenceCounterWrapper(InstanceType* ptr,U&& uref)
  : m_instance(ptr)
  {
    ARCCORE_UNUSED(uref);
    _checkHasReferenceCounterTag();
 }
  explicit ReferenceCounterWrapper(InstanceType* ptr)
  : m_instance(ptr)
  {
    _checkHasReferenceCounterTag();
  }
  //! Autorise à convertir si 'T*' et 'InstanceType*' sont convertibles
  template<typename T,
           typename X = typename std::is_convertible<T*,InstanceType*>::type>
	explicit ReferenceCounterWrapper(const ReferenceCounterWrapper<T>& r)
  : m_instance(r.get())
  {
    _checkHasReferenceCounterTag();
  }
  ReferenceCounterWrapper() = default;
 public:
  InstanceType* get() const { return m_instance.get(); }
  void reset() { m_instance = nullptr; }
 private:
  Arccore::ReferenceCounter<InstanceType> m_instance;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Spécialisation pour indiquer qu'on utilise l'implémentation 'shared_ptr'
template<typename InstanceType>
struct RefTraitsTagId<InstanceType,REF_TAG_SHARED_PTR>
{
  typedef std::shared_ptr<InstanceType> ImplType;
  static constexpr int RefType = REF_TAG_SHARED_PTR;
};

//! Spécialisation pour indiquer qu'on utilise l'implémentation 'ReferenceCounter'
template<typename InstanceType>
struct RefTraitsTagId<InstanceType,REF_TAG_REFERENCE_COUNTER>
{
  typedef Internal::ReferenceCounterWrapper<InstanceType> ImplType;
  static constexpr int RefType = REF_TAG_REFERENCE_COUNTER;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence à une instance.
 *
 * Cette classe utilise un compteur de référence pour gérer la durée de vie
 * d'une instance C++. Elle fonctionne de manière similaire à std::shared_ptr.
 *
 * Lorsque la dernière instance de cette classe est détruite, l'instance
 * référencée est détruite. La manière de détruire l'instance associée
 * est spécifié lors de la création de la première référence via l'appel
 * à une des méthodes create() ou createWithHandle().
 *
 * Il existe deux implémentation possibles pour compter les références.
 * Par défaut, on utilise 'std::shared_ptr'. Il est aussi possible
 * d'utiliser un compteur de référence interne à la classe ce qui
 * permet d'être compatible avec la classe ReferenceCounter et aussi de
 * pouvoir récupérer une référence à partir de l'instance elle même. Cette
 * deuxième implémentation est accessible si le type \a InstanceType
 * définit un type ReferenceCounterTagType valant ReferenceCounterTag.
 */
template<typename InstanceType,int ImplTagId>
class Ref
: public RefBase
{
 public:
  typedef Ref<InstanceType,ImplTagId> ThatClass;
  typedef typename RefTraitsTagId<InstanceType,ImplTagId>::ImplType ImplType;
 private:

  template<typename... _Args>
	using _IsRefConstructible = typename std::enable_if<std::is_constructible<ImplType, _Args...>::value>::type;

  using RefBase::DeleterBase;
  class Deleter : DeleterBase
  {
   public:
    Deleter(Internal::ExternalRef h) : m_handle(std::move(h)), m_no_destroy(false){}
    Deleter(Internal::ExternalRef h,bool no_destroy)
    : m_handle(std::move(h)), m_no_destroy(no_destroy){}
    void operator()(InstanceType* tt)
    {
      if (m_no_destroy)
        return;
      bool is_destroyed = this->_destroyHandle(tt,m_handle);
      if (!is_destroyed)
        delete tt;
    }
    void setNoDestroy(bool x) { m_no_destroy = x; }
   private:
    Internal::ExternalRef m_handle;
    bool m_no_destroy;
  };
 public:
  static constexpr int RefType = RefTraitsTagId<InstanceType,ImplTagId>::RefType;
 private:
  explicit Ref(InstanceType* t) : m_instance(t,Deleter(nullptr)){}
  Ref(InstanceType* t,Internal::ExternalRef handle) : m_instance(t,Deleter(handle)){}
  Ref(InstanceType* t,bool no_destroy) : m_instance(t,Deleter(nullptr,no_destroy)){}
 private:
  Ref(ImplType&& t) : m_instance(t) {}
 public:
  /*!
   * \brief Construit une référence issue d'une autre référence sur un type compatible.
   *
   * La conversion est autorisée si on peut construire une instance de 'ImplType'
   * à partir de celle de celle de Ref<T>::ImplType.
   */
  template<typename T, typename = _IsRefConstructible<typename Ref<T>::ImplType> >
	Ref(const Ref<T>& rhs) noexcept
  : m_instance(rhs._internalInstance()){}
  Ref() = default;
  Ref(const ThatClass& rhs) = default;
  ThatClass& operator=(const ThatClass& rhs) = default;
  ~Ref() = default;
  
 public:
  /*!
   * \internal
   * \brief Créé une référence à partir de l'instance \a t.
   *
   * Cette méthode est interne à %Arccore.
   *
   * L'instance \a t doit avoir été créée par l'opérateur 'operator new'
   * et sera détruite par l'opérateur 'operator delete'
   */
  static ThatClass create(InstanceType* t)
  {
    return ThatClass(t);
  }

  template<typename PointerType, typename... Args>
  static inline Ref<InstanceType>
  createRef(Args&&... args)
  {
    PointerType* pt = new PointerType(std::forward<Args>(args)...);
    return Ref<InstanceType>(pt);
  }

  /*!
   * \internal
   * \brief Créé une référence à partir d'une instance ayant une
   * référence externe.
   */
  static ThatClass createWithHandle(InstanceType* t,Internal::ExternalRef handle)
  {
    return ThatClass(t,handle);
  }

  static ThatClass _createNoDestroy(InstanceType* t)
  {
    return ThatClass(t,true);
  }

 public:
  //! Instance associée ou `nullptr` si aucune
  InstanceType* get() const { return m_instance.get(); }
  //! Indique si le compteur référence une instance non nulle.
  bool isNull() const { return m_instance.get()==nullptr; }
  InstanceType* operator->() const { return m_instance.get(); }
  //! Positionne l'instance au pointeur nul.
  void reset() { m_instance.reset(); }
  /*!
   * \internal
   * \brief Libère le pointeur du compteur de référence sans le détruire.
   * Cette méthode n'est accessible que pour les références utisant std::shared_ptr.
   */
  InstanceType* _release()
  {
    // Relâche l'instance. Pour cela, on indique au destructeur
    // de ne pas détruire l'instance de 'm_instance' et on
    // retourne cette dernière.
    Deleter* r = std::get_deleter<Deleter>(m_instance);
    if (r)
      r->setNoDestroy(true);
    InstanceType* t = m_instance.get();
    m_instance.reset();
    return t;
  }
  const ImplType& _internalInstance() const { return m_instance; }
 private:
  ImplType m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename InstanceType,int TagId>
inline bool operator==(const Ref<InstanceType,TagId>& a,const Ref<InstanceType,TagId>& b)
{
  return a.get()==b.get();
}

template<typename InstanceType,int TagId>
inline bool operator!=(const Ref<InstanceType,TagId>& a,const Ref<InstanceType,TagId>& b)
{
  return a.get()!=b.get();
}

template<typename InstanceType,int TagId>
inline bool operator<(const Ref<InstanceType,TagId>& a,const Ref<InstanceType,TagId>& b)
{
  return a.get()<b.get();
}

template<typename InstanceType,int TagId>
inline bool operator!(const Ref<InstanceType,TagId>& a)
{
  return a.isNull();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une référence sur un pointeur.
 *
 * La pointeur \a t doit avoir été alloué par l'operateur 'operator new' et
 * sera détruit par l'opérateur 'operator delete' lorsqu'il n'y aura plus
 * de référence dessus.
 */
template<typename InstanceType> auto
makeRef(InstanceType* t) -> Ref<InstanceType> 
{
  return Ref<InstanceType>::create(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère une référence sur le pointeur \a t.
 *
 * Cette méthode n'est disponible que si la classe InstanceType utilise un
 * compteur de réference (ImplTagId==REF_TAG_REFERENCE_COUNTER).
 *
 * \code
 * class A {};
 * class B : public A {};
 * Ref<B> rb = ...;
 * B* b = rb.get();
 * Ref<A> ra = makeRefFromInstance<A>(b);
 * \endcode
 */
template<typename InstanceType,
         typename InstanceType2,
         typename std::enable_if_t<Ref<InstanceType>::RefType,int> = REF_TAG_REFERENCE_COUNTER >
inline Ref<InstanceType>
makeRefFromInstance(InstanceType2* t)
{
  return Ref<InstanceType>::create(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

