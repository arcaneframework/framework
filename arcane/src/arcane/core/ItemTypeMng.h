// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeMng.h                                               (C) 2000-2026 */
/*                                                                           */
/* Mesh entity type manager.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEMNG_H
#define ARCANE_CORE_ITEMTYPEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ItemTypes.h"

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
 * \brief Mesh entity type manager.
 *
 * You must call build(IMesh*) before being able to use this
 * instance.
 *
 * The desired types (other than the default types) must be added
 * before the first mesh is created. It is not possible
 * to create new types during execution.
 * 
 * The available types must be strictly identical for all
 * processes (i.e. All ItemTypeMngs of all processes must
 * have the same types).
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

  //! Empty constructor (uninitialized)
  ItemTypeMng();
  ~ItemTypeMng();

 public:

  /*!
   * \brief Effective constructor.
   *
   * This method should only be called before initializing
   * the obsolete singleton instance.
   *
   * \deprecated Use build(IMesh*) instead.
   */
  ARCCORE_DEPRECATED_REASON("Y2025: Use build(IMesh*) instead")
  void build(IParallelSuperMng* parallel_mng, ITraceMng* trace);

  /*!
   * \brief Constructs the instance associated with the mesh \a mesh.
   */
  void build(IMesh* mesh);

 private:

  /*! \brief Singleton instance of the type
   *
   * The singleton is created upon the first call to this function.
   * It remains valid until destroySingleton() has been called
   *
   * \todo: to be removed as soon as no one accesses singleton()
   */
  static ItemTypeMng* _singleton();

  /*!
   * \brief Destroys the singleton
   *
   * The singleton can then be reconstructed by calling destroySingleton()
   */
  static void _destroySingleton();

  static String _legacyTypeName(Integer t);

 public:

  /*!
   * \brief Singleton instance of the type
   *
   * The singleton is created upon the first call to this function.
   * It remains valid until destroySingleton() has been called
   */
  ARCCORE_DEPRECATED_2021("Use IMesh::itemTypeMng() to get an instance of ItemTypeMng")
  static ItemTypeMng* singleton() { return _singleton(); }

  /*!
   * \brief Destroys the singleton
   *
   * The singleton can then be reconstructed by calling singleton()
   */
  ARCCORE_DEPRECATED_2021("Do not use this method")
  static void destroySingleton() { _destroySingleton(); }

 public:

  //! List of available types
  ConstArrayView<ItemTypeInfo*> types() const;

  //! Type corresponding to the number \a id
  ItemTypeInfo* typeFromId(Integer id) const;

  //! Type corresponding to the number \a id
  ItemTypeInfo* typeFromId(ItemTypeId id) const;

  //! Name of the type corresponding to the number \a id
  String typeName(Integer id) const;

  //! Name of the type corresponding to the number \a id
  String typeName(ItemTypeId id) const;

  //! Prints information about available types to the stream \a ostr
  void printTypes(std::ostream& ostr);

  //! Indicates if the mesh \a mesh contains generic cells (outside of built-in or additional types)
  bool hasGeneralCells(IMesh* mesh) const;

  //! Allows the mesh to indicate to the ItemTypeMng if it has generic cells
  void setMeshWithGeneralCells(IMesh* mesh) noexcept;

  /*!
   * \brief Builds types to manage polygons.
   *
   * This makes the ITI_GenericPolygon type accessible.
   * If these types have already been built, this method has no effect.
   */
  void buildPolygonTypes();

  /*!
   * \brief Returns the type for a polygon having \a nb_node.
   *
   * If \a nb_node is between 3 and 8 inclusive, it returns the corresponding internal type
   * (ITI_Triangle3, ITI_Quad4, ..., ITI_Octogon8).
   * Otherwise, it returns the additional type provided that buildPolygonTypes()
   * has been called beforehand.
   *
   * Throws a NotSupportedException if no type matches.
   */
  ItemTypeId getPolygonType(Int16 nb_node) const;

  //! number of available types
  static Integer nbBasicItemType();

  //! number of built-in types (excluding additional types)
  static Integer nbBuiltInItemType();

  // AMR
  static Int32 nbHChildrenByItemType(Integer type);

 private:

  //! Singleton instance
  static ItemTypeMng* singleton_instance;

  //! Number of built-in types (excluding additional types)
  static const Integer m_nb_builtin_item_type;

  //! Initialization flag
  bool m_initialized = false;

  std::atomic<Int32> m_initialized_counter = 0;

  //! Trace manager
  ITraceMng* m_trace = nullptr;

  //! List of types
  UniqueArray<ItemTypeInfo*> m_types;

  //! Allocations of type objects (a pointer is needed to avoid multiple inclusion)
  MultiBufferT<ItemTypeInfoBuilder>* m_types_buffer = nullptr;

  //! Set of meshes containing generic cells (without a defined type)
  std::set<IMesh*> m_mesh_with_general_cells;

  //! Array containing type data.
  UniqueArray<Integer> m_ids_buffer;

  //! Indicates if the types managing polygons have already been built.
  bool m_has_polygon_type = false;

 private:

  void _buildSingleton(IParallelSuperMng* parallel_mng, ITraceMng* trace);
  void _buildTypes(IMesh* mesh, IParallelSuperMng* parallel_mng, ITraceMng* trace);
  //! Reads types from a file named filename
  void _readTypes(IParallelSuperMng* parallel_mng, const String& filename);
  void _addPolygonType(Int16 type_id,Int32 nb_node,const String& type_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
