// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleHTMLMeshAMRPatchExporter.cc                           (C) 2000-2026 */
/*                                                                           */
/* HTML mesh writer, with SVG.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This must be put first for MSVC otherwise we won't have 'M_PI'
#define _USE_MATH_DEFINES
#include <cmath>

#include "arcane/cartesianmesh/SimpleHTMLMeshAMRPatchExporter.h"

#include "arcane/cartesianmesh/AMRPatchPosition.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypes.h"

#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using std::ostream;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleHTMLMeshAMRPatchExporter::Impl
{
 public:

  Impl() = default;
  void addPatch(const CartesianPatch& patch);
  void write(ostream& ofile);

 private:

  void _writeJs(ostream& ofile);
  void _writeCss(ostream& ofile);
  void _writeBodyHtml(ostream& ofile);
  void _writeHeaderSvg(const CartesianPatch& ground_patch);
  void _writeText(Real x, Real y, StringView color, StringView text, Real rotation, bool do_background);
  void _writePatch(const CartesianPatch& patch);

 private:

  Real m_font_size = 0;
  bool m_has_ground = false;
  bool m_has_patch = false;
  StringBuilder m_header_svg{};
  StringBuilder m_patches{};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
addPatch(const CartesianPatch& patch)
{
  if (patch.position().isNull()) {
    return;
  }
  if (patch.level() == 0) {
    if (m_has_ground) {
      ARCANE_FATAL("Ground patch already wrote");
    }
    _writeHeaderSvg(patch);
    m_has_ground = true;
  }
  Int32 level = patch.level();
  Int32 index = patch.index();

  m_patches += String::format("<g class='level-{0}' id='patch-{1}'>\n", level, index);
  _writePatch(patch);
  m_patches += "</g>\n";
  m_has_patch = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
write(ostream& ofile)
{
  if (!m_has_patch) {
    return;
  }
  if (!m_has_ground) {
    ARCANE_FATAL("Need ground patch to write");
  }
  ofile << "<!DOCTYPE html>\n";
  ofile << "<html>\n";

  ofile << "<head>\n";
  _writeJs(ofile);
  _writeCss(ofile);
  ofile << "</head>\n";

  ofile << "<body>\n";
  _writeBodyHtml(ofile);
  ofile << "</body>\n";

  ofile << "</html>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writeJs(ostream& ofile)
{
  ofile << "<script>\n";
  //TODO
  ofile << "</script>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writeCss(ostream& ofile)
{
  ofile << "<style>\n";

  ofile << ".uid-text {font-size: " << m_font_size << "px;}\n";

  ofile << "</style>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writeBodyHtml(ostream& ofile)
{
  ofile << m_header_svg;
  ofile << m_patches;
  ofile << "</g>\n";
  ofile << "</svg>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writeHeaderSvg(const CartesianPatch& ground_patch)
{
  CellGroup cells = ground_patch.patchInterface()->cells();
  if (ground_patch.level() != 0) {
    ARCANE_FATAL("Not ground patch");
  }
  if (cells.null()) {
    ARCANE_FATAL("Empty patch");
  }
  IMesh* mesh = cells.mesh();
  Int32 mesh_dim = mesh->dimension();
  if (mesh_dim != 2)
    ARCANE_FATAL("Invalid dimension ({0}) for mesh. Only 2D mesh is allowed", mesh_dim);

  // Note: since by default in SVG the origin is top left, we take for each
  // 'Y' value its opposite for display.
  // NOTE: we could use SVG transformations but it is more complicated to
  // handle for text display

  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Real mul_value = 1000.0;
  const Real min_val = std::numeric_limits<Real>::lowest();
  const Real max_val = std::numeric_limits<Real>::max();
  Real2 min_bbox(max_val, max_val);
  Real2 max_bbox(min_val, min_val);
  // Calculate the center of the cells and the bounding box of the cell group.
  std::map<Int32, Real2> cells_center;
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Real3 center;
    Integer nb_node = cell.nbNode();
    if (nb_node > 0) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real3 node_coord_3d = nodes_coord[cell.node(i)] * mul_value;
        Real2 node_coord(node_coord_3d.x, -node_coord_3d.y);
        min_bbox = math::min(min_bbox, node_coord);
        max_bbox = math::max(max_bbox, node_coord);
        center += node_coord_3d;
      }
      center /= nb_node;
    }
    Real2 center_2d(center.x, -center.y);
    cells_center[cell.localId()] = center_2d;
  }

  Real bbox_width = math::abs(max_bbox.x - min_bbox.x);
  Real bbox_height = math::abs(max_bbox.y - min_bbox.y);
  Real max_dim = math::max(bbox_width, bbox_height);
  m_font_size = max_dim / 80.0;

  // Add 10% of the patch dimensions on both sides of the viewBox to ensure
  // that the text is properly written (as it might overflow the bounding box)
  m_header_svg += "<?xml version=\"1.0\"?>\n";
  m_header_svg += String::format("<svg viewBox='{0},{1},{2},{3}' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n",
                                 (min_bbox.x - bbox_width * 0.1),
                                 (min_bbox.y - bbox_height * 0.1),
                                 (bbox_width * 1.2),
                                 (bbox_height * 1.2));
  m_header_svg += String::format("<!-- V3 bbox min_x={0} min_y={1} max_x={2} max_y={3} -->",
                                 min_bbox.x,
                                 min_bbox.y,
                                 max_bbox.x,
                                 max_bbox.y);

  m_header_svg += "<title>Mesh</title>\n";
  m_header_svg += "<desc>MeshExample</desc>\n";

  //ofile << "<g transform='matrix(1,0,0,-1,0,200)'>\n";
  //ofile << "<g transform='translate(" << min_bbox.x << "," << -min_bbox.y << ")'>\n";
  m_header_svg += "<g>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writeText(Real x, Real y, StringView color, StringView text, Real rotation, bool do_background)
{
  // Displays a white background below the text.
  if (do_background) {
    m_patches += String::format("<text class='uid-text' x='{0}' y='{1}' dominant-baseline='central' text-anchor='middle' style='stroke:white; stroke-width:0.6em'", x, y);
    if (rotation != 0.0) {
      m_patches += String::format(" transform='rotate({0}, {1}, {2})'", rotation, x, y);
    }
    m_patches += String::format(">{0}</text>\n", text);
  }

  m_patches += String::format("<text class='uid-text' x='{0}' y='{1}' dominant-baseline='central' text-anchor='middle' fill='{2}'", x, y, color);

  if (rotation != 0.0) {
    m_patches += String::format(" transform='rotate({0}, {1}, {2})'", rotation, x, y);
  }
  m_patches += String::format(">{0}</text>\n", text);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::Impl::
_writePatch(const CartesianPatch& patch)
{
  CellGroup cells = patch.patchInterface()->cells();
  if (cells.null())
    return;
  IMesh* mesh = cells.mesh();
  Int32 mesh_dim = mesh->dimension();
  if (mesh_dim != 2)
    ARCANE_FATAL("Invalid dimension ({0}) for mesh. Only 2D mesh is allowed", mesh_dim);

  // Note: since by default in SVG the origin is top left, we take for each
  // 'Y' value its opposite for display.
  // NOTE: we could use SVG transformations but it is more complicated to
  // handle for text display

  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Real mul_value = 1000.0;
  const Real min_val = std::numeric_limits<Real>::lowest();
  const Real max_val = std::numeric_limits<Real>::max();
  Real2 min_bbox(max_val, max_val);
  Real2 max_bbox(min_val, min_val);

  // Calculate the center of the cells and the bounding box of the cell group.
  std::map<Int32, Real2> cells_center;
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Real3 center;
    Integer nb_node = cell.nbNode();
    if (nb_node > 0) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real3 node_coord_3d = nodes_coord[cell.node(i)] * mul_value;
        Real2 node_coord(node_coord_3d.x, -node_coord_3d.y);
        min_bbox = math::min(min_bbox, node_coord);
        max_bbox = math::max(max_bbox, node_coord);
        center += node_coord_3d;
      }
      center /= nb_node;
    }
    Real2 center_2d(center.x, -center.y);
    cells_center[cell.localId()] = center_2d;
  }

  // Display the contour and uniqueId() for each cell.
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Real2 cell_pos = cells_center[cell.localId()];
    Integer nb_node = cell.nbNode();
    m_patches += "<path d='";
    nb_node = cell.typeInfo()->linearTypeInfo()->nbLocalNode();
    for (Integer i = 0; i < nb_node; ++i) {
      Real3 node_coord_3d = nodes_coord[cell.node(i)];
      Real2 node_coord(node_coord_3d.x, -node_coord_3d.y);
      node_coord *= mul_value;
      if (i == 0)
        m_patches += "M ";
      else
        m_patches += "L ";
      // performs a homothety to clearly see the faces in case of welding.
      Real2 coord = cell_pos + (node_coord - cell_pos) * 0.98;
      m_patches += String::format("{0} {1} ", coord.x, coord.y);
    }
    m_patches += "z'";
    if (icell->hasFlags(ItemFlags::II_Overlap) && icell->hasFlags(ItemFlags::II_InPatch)) {
      if (cell.isOwn())
        m_patches += " fill='magenta'";
      else
        m_patches += " fill='darkmagenta'";
    }
    if (icell->hasFlags(ItemFlags::II_Overlap)) {
      if (cell.isOwn())
        m_patches += " fill='orange'";
      else
        m_patches += " fill='darkorange'";
    }
    else {
      if (cell.isOwn())
        m_patches += " fill='yellow'";
      else
        m_patches += " fill='gold'";
    }
    m_patches += " stroke='black'";
    m_patches += " stroke-width='1'/>\n";
    _writeText(cell_pos.x, cell_pos.y, "blue", String::fromNumber(cell.uniqueId().asInt64()), 0.0, false);
  }

  // Display the uniqueId() for each node.
  {
    // Set of nodes already processed to display them only once.
    std::set<Int32> nodes_done;
    ENUMERATE_CELL (icell, cells) {
      Cell cell = *icell;
      Integer nb_node = cell.nbNode();
      for (Integer i = 0; i < nb_node; ++i) {
        Node node = cell.node(i);
        Int32 lid = node.localId();
        if (nodes_done.find(lid) != nodes_done.end())
          continue;
        nodes_done.insert(lid);
        Real3 coord_3d = nodes_coord[node];
        Real2 coord(coord_3d.x, -coord_3d.y);
        coord *= mul_value;
        _writeText(coord.x, coord.y, "green", String::fromNumber(node.uniqueId().asInt64()), 0.0, true);
      }
    }
  }

  // Display the uniqueId() for each face.
  // Performs a possible rotation so that the display of the face number is aligned
  // with its segment.
  {
    // Set of faces already processed to display them only once.
    std::set<Int32> faces_done;
    ENUMERATE_CELL (icell, cells) {
      Cell cell = *icell;
      Integer nb_face = cell.nbFace();
      for (Integer i = 0; i < nb_face; ++i) {
        Face face = cell.face(i);
        Int32 lid = face.localId();
        if (faces_done.find(lid) != faces_done.end())
          continue;
        faces_done.insert(lid);
        // In case of multi-dimensional mesh, it is possible
        // to have faces reduced to a point.
        if (face.nbNode() < 2)
          continue;
        Real3 node0_coord = nodes_coord[face.node(0)];
        Real3 node1_coord = nodes_coord[face.node(1)];
        Real3 face_coord_3d = (node0_coord + node1_coord) / 2.0;

        Real2 face_coord(face_coord_3d.x, -face_coord_3d.y);
        face_coord *= mul_value;
        Real3 direction = node1_coord - node0_coord;
        direction = math::mutableNormalize(direction);
        // TODO: check between -1.0 and 1.0
        // Calculates the rotation angle so that the display of the face number is aligned with
        // the edge line of the face.
        double angle = math::abs(std::asin(direction.y)) / M_PI * 180.0;
        Real2 cell_center = cells_center[cell.localId()];
        Real2 coord = cell_center + (face_coord - cell_center) * 0.92;
        _writeText(coord.x, coord.y, "red", String::fromNumber(face.uniqueId().asInt64()), angle, true);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHTMLMeshAMRPatchExporter::
SimpleHTMLMeshAMRPatchExporter()
: m_p(new Impl())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleHTMLMeshAMRPatchExporter::
~SimpleHTMLMeshAMRPatchExporter()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::
addPatch(const CartesianPatch& patch)
{
  m_p->addPatch(patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleHTMLMeshAMRPatchExporter::
write(std::ostream& ofile)
{
  m_p->write(ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
