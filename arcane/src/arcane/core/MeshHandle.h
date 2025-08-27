// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.h                                                (C) 2000-2023 */
/*                                                                           */
/* Handle sur un maillage.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHHANDLE_H
#define ARCANE_MESHHANDLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/StringView.h"
#include "arccore/base/String.h"

#include "arcane/utils/UtilsTypes.h"
#include "arcane/core/ArcaneTypes.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Handle sur un maillage.
 *
 * Cette classe utilise la sémantique d'un compteur de référence.
 *
 * Cette classe permet de gérer une référence à un maillage (IMesh) avant
 * qu'il ne soit explicitement créé. Cela permet aux services et modules
 * de spécifier lors de leur construction à quel maillage ils font référence.
 *
 * Elle permet aussi d'associer des données utilisateurs au maillage
 * via meshUserDataList().
 */
class ARCANE_CORE_EXPORT MeshHandle
{
 private:

  // Temporaire: pour accéder au constructeur qui utilise ISubDomain.
  friend class MeshMng;

 private:

  class MeshHandleRef
  : public Arccore::ReferenceCounterImpl
  {
   public:

    MeshHandleRef()
    : m_is_null(true)
    {}
    MeshHandleRef(ISubDomain* sd, const String& name);
    ~MeshHandleRef();

   public:

    const String& meshName() const { return m_mesh_name; }
    bool isNull() const { return m_is_null; }
    IMesh* mesh() const { return m_mesh_ptr; }
    IMeshBase* meshBase() const { return m_mesh_base_ptr; }
    ISubDomain* subDomain() const { return m_sub_domain; }
    IMeshMng* meshMng() const { return m_mesh_mng; }
    ITraceMng* traceMng() const { return m_trace_mng; }
    IVariableMng* variableMng() const { return m_variable_mng; }
    IUserDataList* userDataList() const { return m_user_data_list; }
    Observable* onDestroyObservable() const { return m_on_destroy_observable; }
    bool isDoFatalInMeshMethod() const { return m_do_fatal_in_mesh_method; }

   public:

    void _destroyMesh();
    void _setMesh(IMesh* mesh);

   private:

    String m_mesh_name;
    IMesh* m_mesh_ptr = nullptr;
    IMeshBase* m_mesh_base_ptr = nullptr;
    // Pour l'instant on en a besoin mais il faudrait le supprimer
    ISubDomain* m_sub_domain = nullptr;
    IUserDataList* m_user_data_list = nullptr;
    IMeshMng* m_mesh_mng = nullptr;
    ITraceMng* m_trace_mng = nullptr;
    IVariableMng* m_variable_mng = nullptr;
    Observable* m_on_destroy_observable = nullptr;
    bool m_is_null = true;
    bool m_do_fatal_in_mesh_method = false;
  };

 public:

  MeshHandle();

 private:

  // TODO rendre accessible uniquement aux classes implémentant IMeshMng.
  MeshHandle(ISubDomain* sd, const String& name);

 public:

  /*!
   * \brief Maillage associé.
   *
   * Il est interdit d'appeler cette méthode si le maillage n'a pas encore été
   * créé. A terme, une exception sera levée dans ce cas.
   *
   * Si on n'est pas certain que le maillage existe, on peut tester son
   * existence via hasMesh().
   *
   * \pre hasMesh() == true
   */
  IMesh* mesh() const;

  //! Indique si le maillage associé a déjà été créé (i.e: mesh() est valide)
  bool hasMesh() const;

  /*!
   * \brief Retourne le maillage associé à cette instance.
   *
   * Contrairement à mesh(), cette peut-être appelée si le maillage associé n'a pas
   * encore été créé. Dans ce cas on retourne un pointeur nul.
   */
  IMesh* meshOrNull() const;

 public:

  //! Sous-domaine associé. Null si isNull() est vrai.
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get ISubDomain from another way")
  ISubDomain* subDomain() const { return m_ref->subDomain(); }

 public:

  //! Gestionnaire de maillage associé. nullptr si isNull() est vrai.
  IMeshMng* meshMng() const;

  //! Gestionnaire de trace associé. nullptr si isNull() est vrai.
  ITraceMng* traceMng() const;

  //! Gestionnaire de variable associé. nullptr si isNull() est vrai.
  IVariableMng* variableMng() const;

  //! Application associée. nullptr si isNull() est vrai.
  IApplication* application() const;

  //! Données utilisateurs associées
  IUserDataList* meshUserDataList() const { return m_ref->userDataList(); }

  const String& meshName() const { return m_ref->meshName(); }

  //! Indique si le handle est nul (il ne référence aucun maillage existant ou non)
  bool isNull() const { return m_ref->isNull(); }

  //! Observable pour être notifié de la destruction
  IObservable* onDestroyObservable() const;

  //! \internal
  const void* reference() const { return m_ref.get(); }

 public:

  //! \internal
  void _setMesh(IMesh* mesh) { m_ref->_setMesh(mesh); }

  //! \internal
  void _destroyMesh() { m_ref->_destroyMesh(); }

  //! \internal
  IMesh* _internalMeshOrNull() const { return m_ref->mesh(); }

 private:

  Arccore::ReferenceCounter<MeshHandleRef> m_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de compatibilité pour contenir un MeshHandle ou un IMesh*.
 *
 * A terme les constructeurs et convertisseurs vers IMesh* seront supprimés
 */
class ARCANE_CORE_EXPORT MeshHandleOrMesh
{
 public:

  // NOTE: Les constructeurs ne doivent pas être explicites
  // pour autoriser les conversions

  //! Construit une instance à partir d'un MeshHandle
  MeshHandleOrMesh(const MeshHandle& handle);

  /*!
   * \brief Construit une instance à partir d'un IMesh*.
   *
   * Si \a mesh est nul, le MeshHandle associé sera aussi nul.
   */
  MeshHandleOrMesh(IMesh* mesh);

  //! Maillage associé. Peut être nul si le maillage n'a pas encore été créé
  IMesh* mesh() const { return m_handle.meshOrNull(); }

  //! Maillage associé. Peut être nul si le maillage n'a pas encore été créé
  operator IMesh*() const { return mesh(); }

  //! handle associé.
  const MeshHandle& handle() const { return m_handle; }

 private:

  MeshHandle m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
