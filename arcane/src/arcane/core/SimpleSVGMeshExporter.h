// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleSVGMeshExporter.h                                     (C) 2000-2025 */
/*                                                                           */
/* SVG mesh writer.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMPLESVGMESHEXPORTER_H
#define ARCANE_CORE_SIMPLESVGMESHEXPORTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Exporting an SVG mesh.
 *
 * This class only works for 2D meshes. Only the `x` and `y` components
 * are considered. After creating an instance, it is possible
 * to call the writeGroup() method to export the entities associated with the mesh
 * group (nodes, faces, and cells).
 */
class ARCANE_CORE_EXPORT SimpleSVGMeshExporter
{
  class Impl;

 public:

  /*!
  * \brief Create an instance associated with the \a ofile stream.
  *
  * The \a ofile stream must remain valid throughout the lifetime of the instance.
  */
  SimpleSVGMeshExporter(std::ostream& ofile);
  SimpleSVGMeshExporter(const SimpleSVGMeshExporter& rhs) = delete;
  SimpleSVGMeshExporter& operator=(const SimpleSVGMeshExporter& rhs) = delete;
  ~SimpleSVGMeshExporter();

  /*!
   * \brief Exports the entities of the \a cells group.
   *
   * Currently, it is not possible to call this method multiple times.
   */
  void write(const CellGroup& cells);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
