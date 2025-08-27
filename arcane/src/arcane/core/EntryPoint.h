// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EntryPoint.h                                                (C) 2000-2022 */
/*                                                                           */
/* Point d'entrée d'un module.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ENTRYPOINT_H
#define ARCANE_ENTRYPOINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/FunctorWithAddress.h"
#include "arcane/core/IEntryPoint.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour construire un point d'entrée.
 *
 * Normalement cette classe n'est pas utilisée directement. Pour
 * construire un point d'entrée, il faut utiliser addEntryPoint().
 */
class ARCANE_CORE_EXPORT EntryPointBuildInfo
{
 public:
  /*!
   * \brief Informations de construction d'un point d'entrée.
   *
   * \param module module associé à la fonction
   * \param where endroit de la boucle en temps où est appelé le point d'entrée
   * \param property propriétés du point d'entrée (voir IEntryPoint)
   * \param name nom du point d'entrée
   * \param caller encapsulation de la méthode à appeler.
   * \param is_destroy_caller indique si le point d'entrée doit détruire
   * le fonctor \a caller.
   *
   * En général, \a is_destroy_caller doit valoir \a true sinon la
   * mémoire ne sera pas libéré. A noter que le wrapping C# gère le fonctor
   * via un garbage collector et donc dans ce cas \a is_destroy_caller doit
   * valoir \a false.
   */
  EntryPointBuildInfo(IModule* module,const String& name,
                      IFunctor* caller,const String& where,int property,
                      bool is_destroy_caller)
  : m_module(module), m_name(name), m_caller(caller), m_where(where),
    m_property(property), m_is_destroy_caller(is_destroy_caller)
  {
  }

 public:

  IModule* module() const { return m_module; }
  const String& name() const { return m_name; }
  IFunctor* caller() const { return m_caller; }
  const String& where () const { return m_where; }
  int property() const { return m_property; }
  bool isDestroyCaller() const { return m_is_destroy_caller; }

 private:

  IModule* m_module;
  String m_name;
  IFunctor* m_caller;
  String m_where;
  int m_property;
  bool m_is_destroy_caller;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Point d'entrée d'un module.
 */
class ARCANE_CORE_EXPORT EntryPoint
: public IEntryPoint
{
 public:

  //! Libère les ressources
  ~EntryPoint() override;

 public:

  /*!
   * \brief Construit et retourne un point d'entrée.
   *
   * Le point d'entrée est construit avec les informations données par \bi.
   * Il est automatiquement ajouté au gestionnaire IEntryPointMng et ne doit
   * pas être détruit explicitement.
   */
  static EntryPoint* create(const EntryPointBuildInfo& bi);

 public:
	
  String name() const override { return m_name; }
  String fullName() const override { return m_full_name; }
  ISubDomain* subDomain() const override { return m_sub_domain; }
  IModule* module() const override { return m_module; }
  void executeEntryPoint() override;
  Real totalCPUTime() const override;
  Real lastCPUTime() const override;
  Real totalElapsedTime() const override;
  Real lastElapsedTime() const override;
  Real totalTime(Timer::eTimerType type) const override;
  Real lastTime(Timer::eTimerType type) const override;
  Integer nbCall() const override { return m_nb_call; }
  String where() const override { return m_where; }
  int property() const override { return m_property; }

 private:

  ISubDomain* m_sub_domain = nullptr; //!< Gestionnaire de sous-domaine
  IFunctor* m_caller = nullptr; //!< Point d'appel
  Timer* m_elapsed_timer = nullptr; //!< Timer horloge du point d'entrée
  String m_name; //!< Nom du point d'entrée
  String m_full_name; //!< Nom du point d'entrée
  IModule* m_module = nullptr; //!< Module associé
  String m_where; //!< Endroit de l'appel
  int m_property = 0; //!< Propriétés du point d'entrée
  Integer m_nb_call = 0; //!< Nombre de fois que le point d'entrée a été exécuté
  bool m_is_destroy_caller = false; //!< Indique si on doit détruire le functor d'appel.

 private:

  explicit EntryPoint(const EntryPointBuildInfo& build_info);

 public:

  EntryPoint(const EntryPoint&) =delete;
  void operator=(const EntryPoint&) =delete;

 private:

  void _getAddressForHyoda(void* =nullptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Routine template permettant de référencer un point d'entrée
 * dans un module.
 *
 * Le paramètre \a ModuleType doit être un type qui dérive de IModule.
 *
 * \param module Module associé à la fonction
 * \param func méthode membre appelée par la fonction
 * \param where endroit ou est appelé le point d'entrée
 * \param property propriétés du point d'entrée (voir IEntryPoint)
 * \param name nom de la fonction pour Arcane
 */
template<typename ModuleType> inline void
addEntryPoint(ModuleType* module,const char* name,void (ModuleType::*func)(),
              const String& where = IEntryPoint::WComputeLoop,
              int property = IEntryPoint::PNone)
{
  IFunctorWithAddress* caller = new FunctorWithAddressT<ModuleType>(module,func);
  EntryPoint::create(EntryPointBuildInfo(module,name,caller,where,property,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Routine template permettant de référencer un point d'entrée
 * dans un module.
 *
 * Le paramètre \a ModuleType doit être un type qui dérive de IModule.
 *
 * \param module Module associé à la fonction
 * \param func méthode membre appelée par la fonction
 * \param where endroit ou est appelé le point d'entrée
 * \param property propriétés du point d'entrée (voir IEntryPoint)
 * \param name nom de la fonction pour Arcane
 */
template<typename ModuleType> inline void
addEntryPoint(ModuleType* module,const String& name,void (ModuleType::*func)(),
              const String& where = IEntryPoint::WComputeLoop,
              int property = IEntryPoint::PNone)
{
  IFunctorWithAddress* caller = new FunctorWithAddressT<ModuleType>(module,func);
  EntryPoint::create(EntryPointBuildInfo(module,name,caller,where,property,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

