// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeMng.h                                               (C) 2000-2022 */
/*                                                                           */
/* Gestionnaire des types d'entité du maillage.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMTYPEMNG_H
#define ARCANE_ITEMTYPEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include "arcane/ItemTypes.h"

#include <set>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace mesh
{
  // TEMPORAIRE: pour que ces classes aient accès au singleton.
  class DynamicMesh;
  class PolyhedralMesh;
} // namespace mesh
class ArcaneMain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemTypeInfo;
class ItemTypeInfoBuilder;
class IParallelSuperMng;
template <class T>
class MultiBufferT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Gestionnaire des types d'entités de maillage.

 Il n'existe qu'une seule instance de ce gestionnaire (singleton).

 Les types souhaités (autre que les types par défaut) doivent être ajoutés
 avant que le premier maillage ne soit créé. Il n'est pas possible
 de créer de nouveaux types pendant l'exécution.

 Les types disponibles doivent être strictement identiques pour tous
 les processus (i.e Tous les ItemTypeMng de tous les processus doivent
 avoir les mêmes types).

 \todo Ajouter les méthodes pour ajouter un nouveau type
 */
class ARCANE_CORE_EXPORT ItemTypeMng
{
  // Ces classes sont ici temporairement tant que le singleton est accessible.
  friend class mesh::DynamicMesh;
  friend class mesh::PolyhedralMesh;
  friend class Application;
  friend class ArcaneMain;
  friend class Item;
  friend ItemTypeInfo;
  friend ItemTypeInfoBuilder;

 protected:

  //! Constructeur vide (non initialisé)
  ItemTypeMng();
  ~ItemTypeMng();

 public:

  //! Constructeur effectif
  void build(IParallelSuperMng* parallel_mng, ITraceMng* trace);

 private:

  /*! \brief Instance singleton du type
   *
   * Le singleton est créé lors du premier appel à cette fonction.
   * Il reste valide tant que destroySingleton() n'a pas été appelé
   *
   * \todo: a supprimer dès que plus personne ne fera d'accès à singleton()
   */
  static ItemTypeMng* _singleton();

  /*!
   * \brief Détruit le singleton
   *
   * Le singleton peut ensuite être reconstruit par appel à destroySingleton()
   */
  static void _destroySingleton();

  static String _legacyTypeName(Integer t);

 public:

  /*!
   * \brief Instance singleton du type
   *
   * Le singleton est créé lors du premier appel à cette fonction.
   * Il reste valide tant que destroySingleton() n'a pas été appelé
   */
  ARCCORE_DEPRECATED_2021("Use IMesh::itemTypeMng() to get an instance of ItemTypeMng")
  static ItemTypeMng* singleton() { return _singleton(); }

  /*!
   * \brief Détruit le singleton
   *
   * Le singleton peut ensuite être reconstruit par appel à singleton()
   */
  ARCCORE_DEPRECATED_2021("Do not use this method")
  static void destroySingleton() { _destroySingleton(); }

 public:

  //! Liste des types disponibles
  ConstArrayView<ItemTypeInfo*> types() const;

  //! Type correspondant au numéro \a id
  ItemTypeInfo* typeFromId(Integer id) const;

  //! Type correspondant au numéro \a id
  ItemTypeInfo* typeFromId(ItemTypeId id) const;

  //! Nom du type correspondant au numéro \a id
  String typeName(Integer id) const;

  //! Nom du type correspondant au numéro \a id
  String typeName(ItemTypeId id) const;

  //! Affiche les infos sur les types disponibles sur le flot \a ostr
  void printTypes(std::ostream& ostr);

  //! Indique si le maillage \a mesh contient des mailles génériques (en dehors des types intégrés ou additionnels)
  bool hasGeneralCells(IMesh* mesh) const;

  //! Permet au maillage d'indiquer à l'ItemTypeMng s'il a des mailles génériques
  void setMeshWithGeneralCells(IMesh* mesh) noexcept;

  //! nombre de types disponibles
  static Integer nbBasicItemType();

  //! nombre de types intégrés (hors types additionnels)
  static Integer nbBuiltInItemType();

  // AMR
  static Int32 nbHChildrenByItemType(Integer type);

 private:

  //! Lecture des types a partir d'un fichier de nom filename
  void readTypes(IParallelSuperMng* parallel_mng, const String& filename);

 private:

  //! Instance singleton
  static ItemTypeMng* singleton_instance;

  //! Nombre de types intégrés (hors types additionnels)
  static const Integer m_nb_builtin_item_type;

  //! Flag d'initialisation
  bool m_initialized = false;

  std::atomic<Int32> m_initialized_counter;

  //! Gestionnaire de traces
  ITraceMng* m_trace = nullptr;

  //! Liste des types
  UniqueArray<ItemTypeInfo*> m_types;

  //! Allocations des objets de type (il faut un pointeur pour eviter inclusion multiple)
  MultiBufferT<ItemTypeInfoBuilder>* m_types_buffer = nullptr;

  //! Ensemble des maillages contenant des mailles générales (sans type défini)
  std::set<IMesh*> m_mesh_with_general_cells;

  /*!
   * \brief Tampon d'allocation des données de type.
   *
   * Ce champs est public en attendant d'affiner la gestion de l'accès.
   */
  UniqueArray<Integer> m_ids_buffer;

 private:

  void _build(IParallelSuperMng* parallel_mng, ITraceMng* trace);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
