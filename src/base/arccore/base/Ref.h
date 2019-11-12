// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Ref.h                                                       (C) 2000-2019 */
/*                                                                           */
/* Gestion des références sur une instance.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REF_H
#define ARCCORE_BASE_REF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExternalRef.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure servant à tagger les interfaces/classes qui utilisent
 * un compteur de référence interne.
 *
 * Ce tag s'utilise via un typedef comme suit:
 *
 * \code
 * class MyClass
 * {
 *   public:
 *    typedef ReferenceCounterTag ReferenceCounterTagType;
 *   public:
 *    void addReference();
 *    void removeReference();
 * };
 */
struct ReferenceCounterTag {};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une référence à un service.
 */
class ARCCORE_BASE_EXPORT RefBase
{
 protected:
  class DeleterBase
  {
   protected:
    bool _destroyHandle(void* instance,Internal::ExternalRef& handle);
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Internal
{
template<typename InstanceType>
class ReferenceCounterWrapper
{
 public:
  template<typename U>
  ReferenceCounterWrapper(InstanceType* ptr,U&& uref)
  : m_instance(ptr)
  {
    ARCCORE_UNUSED(uref);
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
/*!
 * \brief Caractéristiques pour gérer les compteurs de référence.
 *
 * Par défaut, on utilise comme implémentation la classe std::shared_ptr.
 */
template<typename InstanceType,class T = void>
struct RefTraits
{
  typedef std::shared_ptr<InstanceType> ImplType;
};

/*!
 * \brief Spécialisation de la classe gérant un compteur de référence
 * si la classe utilise le tag 'ReferenceCounterTag'.
 *
 * Dans ce cas, on utilise 'ReferenceCounter' comme implémentation.
 */
template<typename InstanceType>
struct RefTraits<InstanceType,std::enable_if_t<std::is_same_v<typename InstanceType::ReferenceCounterTagType,ReferenceCounterTag>>>
{
  typedef Internal::ReferenceCounterWrapper<InstanceType> ImplType;
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
template<typename InstanceType>
class Ref
: public RefBase
{
  typedef typename RefTraits<InstanceType>::ImplType ImplType;
  using RefBase::DeleterBase;
  class Deleter : DeleterBase
  {
   public:
    Deleter(Internal::ExternalRef h) : m_handle(h), m_no_destroy(false){}
    Deleter(Internal::ExternalRef h,bool no_destroy)
    : m_handle(h), m_no_destroy(no_destroy){}
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
  typedef Ref<InstanceType> ThatClass;
 private:
  explicit Ref(InstanceType* t) : m_instance(t,Deleter(nullptr)){}
  Ref(InstanceType* t,Internal::ExternalRef handle) : m_instance(t,Deleter(handle)){}
  Ref(InstanceType* t,bool no_destroy) : m_instance(t,Deleter(nullptr,no_destroy)){}
 public:
  Ref() = default;
  Ref(const Ref<InstanceType>& rhs) = default;
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
  static Ref<InstanceType> create(InstanceType* t)
  {
    return Ref<InstanceType>(t);
  }

  /*!
   * \internal
   * \brief Créé une référence à partir d'une instance ayant une
   * référence externe.
   */
  static Ref<InstanceType> createWithHandle(InstanceType* t,Internal::ExternalRef handle)
  {
    return Ref<InstanceType>(t,handle);
  }

  static Ref<InstanceType> _createNoDestroy(InstanceType* t)
  {
    return Ref<InstanceType>(t,true);
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
    r->setNoDestroy(true);
    InstanceType* t = m_instance.get();
    m_instance.reset();
    return t;
  }
 private:

  ImplType m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename InstanceType>
inline bool operator==(const Ref<InstanceType>& a,const Ref<InstanceType>& b)
{
  return a.get()==b.get();
}

template<typename InstanceType>
inline bool operator!=(const Ref<InstanceType>& a,const Ref<InstanceType>& b)
{
  return a.get()!=b.get();
}

template<typename InstanceType>
inline bool operator<(const Ref<InstanceType>& a,const Ref<InstanceType>& b)
{
  return a.get()<b.get();
}

template<typename InstanceType>
inline bool operator!(const Ref<InstanceType>& a)
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
template<typename InstanceType>
Ref<InstanceType> makeRef(InstanceType* t)
{
  return Ref<InstanceType>::create(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

