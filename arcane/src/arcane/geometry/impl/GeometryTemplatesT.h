// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCGEOSIM_GEOMETRY_IMPL_GEOMETRYTEMPLATES_H
#define ARCGEOSIM_GEOMETRY_IMPL_GEOMETRYTEMPLATES_H

#include <arcane/IMesh.h>
using namespace Arcane;

#include <arcane/IItemOperationByBasicType.h>

#include <arcane/IItemFamily.h>
#include <arcane/ArcaneVersion.h>
#include <arcane/utils/ITraceMng.h>
#include <arcane/MathUtils.h>
#include <arcane/ItemVectorView.h>

#include "arcane/geometry/impl/ItemGroupGeometryProperty.h"

ARCANE_BEGIN_NAMESPACE
NUMERICS_BEGIN_NAMESPACE

template <typename GeometryT>
class GenericGSInternalUpdater : public IItemOperationByBasicType
{
private:
  GeometryT & m_geom;
  ItemGroupGeometryProperty * m_group_property;
  ITraceMng * m_trace_mng;

public:
  GenericGSInternalUpdater(GeometryT & geom, ITraceMng * traceMng) 
    : m_geom(geom), m_trace_mng(traceMng) { }

  void setGroupProperty(ItemGroupGeometryProperty * group_property)
  {
    m_group_property = group_property;
  }

#define SAVE_PROPERTY(property,type,item,group,expr)                    \
  if (m_group_property->hasProperty((property)))                        \
    {                                                                   \
      ItemGroupGeometryProperty::StorageInfo & storage = m_group_property->storages[property]; \
      if (ContainerAccessorT<type>::getVarContainer(storage))           \
        {                                                               \
          IGeometryMng::type##Variable & mMap = *ContainerAccessorT<type>::getVarContainer(storage); \
          ENUMERATE_ITEMWITHNODES((item), (group)) {                    \
            mMap[*(item)] = (expr);                                     \
          }                                                             \
        }                                                               \
    }
  
  template<typename ComputeLineFunctor>
  void applyLineTemplate(ItemVectorView group)
  {
    // Utilise des tableaux locaux plutot qu'une spécialisation par type de propriété (moins de code, plus de souplesse)
    UniqueArray<Real3> centers(group.size());
    UniqueArray<Real3> orientations(group.size());

    ComputeLineFunctor functor(&m_geom);
    ENUMERATE_ITEMWITHNODES(item, group) {
      functor.computeOrientedMeasureAndCenter(*item,orientations[item.index()],centers[item.index()]);
    }

    SAVE_PROPERTY(IGeometryProperty::PMeasure,Real,item,group,math::normeR3(orientations[item.index()]));
    SAVE_PROPERTY(IGeometryProperty::PLength,Real,item,group,math::normeR3(orientations[item.index()]));
    SAVE_PROPERTY(IGeometryProperty::PArea,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PVolume,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PCenter,Real3,item,group,centers[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PNormal,Real3,item,group,orientations[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PVolumeSurfaceRatio,Real,item,group,0);
  }

  template<typename ComputeSurfaceFunctor>
  void applySurfaceTemplate(ItemVectorView group)
  {
    // Utilise des tableaux locaux plutot qu'une spécialisation par type de propriété (moins de code, plus de souplesse)
    UniqueArray<Real3> centers(group.size());
    UniqueArray<Real3> normals(group.size());

    ComputeSurfaceFunctor functor(&m_geom);
    ENUMERATE_ITEMWITHNODES(item, group) {
      functor.computeOrientedMeasureAndCenter(*item,normals[item.index()],centers[item.index()]);
    }

    SAVE_PROPERTY(IGeometryProperty::PMeasure,Real,item,group,math::normeR3(normals[item.index()]));
    SAVE_PROPERTY(IGeometryProperty::PLength,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PArea,Real,item,group,math::normeR3(normals[item.index()]));
    SAVE_PROPERTY(IGeometryProperty::PVolume,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PCenter,Real3,item,group,centers[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PNormal,Real3,item,group,normals[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PVolumeSurfaceRatio,Real,item,group,0);
  }

  template<typename ComputeVolumeFunctor>
  void applyVolumeTemplate(ItemVectorView group)
  {
    // Utilise des tableaux locaux plutot qu'une spécialisation par type de propriété (moins de code, plus de souplesse)
    UniqueArray<Real3> centers(group.size());
    UniqueArray<Real> volumes(group.size());

    ComputeVolumeFunctor functor(&m_geom);
    ENUMERATE_ITEMWITHNODES(item, group) {
      functor.computeOrientedMeasureAndCenter(*item,volumes[item.index()],centers[item.index()]);
    }

    SAVE_PROPERTY(IGeometryProperty::PMeasure,Real,item,group,volumes[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PLength,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PArea,Real,item,group,0);
    SAVE_PROPERTY(IGeometryProperty::PVolume,Real,item,group,volumes[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PCenter,Real3,item,group,centers[item.index()]);
    SAVE_PROPERTY(IGeometryProperty::PNormal,Real3,item,group,0);

    if (m_group_property->hasProperty((IGeometryProperty::PVolumeSurfaceRatio)))
      {
        UniqueArray<Real> areas(group.size());
        ENUMERATE_ITEMWITHNODES(item, group) {
          functor.computeVolumeArea(*item,areas[item.index()]);
        }
        SAVE_PROPERTY(IGeometryProperty::PVolumeSurfaceRatio,Real,item,group,volumes[item.index()]/areas[item.index()]);
      }
  }

  void applyVertex(ItemVectorView group) { ARCANE_UNUSED(group); }

  void applyLine2(ItemVectorView group) {
    applyLineTemplate<typename GeometryT::ComputeLine2>(group);
  }

  void applyTriangle3(ItemVectorView group) {
    applySurfaceTemplate<typename GeometryT::ComputeTriangle3>(group);
  }

  void applyQuad4(ItemVectorView group) {
    applySurfaceTemplate<typename GeometryT::ComputeQuad4>(group);
  }

  void applyPentagon5(ItemVectorView group) {
    applySurfaceTemplate<typename GeometryT::ComputePentagon5>(group);
  }

  void applyHexagon6(ItemVectorView group) {
    applySurfaceTemplate<typename GeometryT::ComputeHexagon6>(group);
  }

  void applyTetraedron4(ItemVectorView group)
  {
    applyVolumeTemplate<typename GeometryT::ComputeTetraedron4>(group);
  }

  void applyPyramid5(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputePyramid5>(group);
  }

  void applyPentaedron6(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputePentaedron6>(group);
  }

  void applyHexaedron8(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeHexaedron8>(group);
  }

  void applyHeptaedron10(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeHeptaedron10>(group);
  }

  void applyOctaedron12(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeOctaedron12>(group);
  }

  void applyHemiHexa7(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeHemiHexa7>(group);
  }

  void applyHemiHexa6(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeHemiHexa6>(group);
  }

  void applyHemiHexa5(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeHemiHexa5>(group);
  }

  void applyAntiWedgeLeft6(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeAntiWedgeLeft6>(group);
  }

  void applyAntiWedgeRight6(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeAntiWedgeRight6>(group);
  }

  void applyDiTetra5(ItemVectorView group) {
    applyVolumeTemplate<typename GeometryT::ComputeDiTetra5>(group);
  }

  void applyDualNode(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyDualEdge(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyDualFace(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyDualCell(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyLine3(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyLine4(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyLine5(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyLine9(ItemVectorView group) { ARCANE_UNUSED(group); }
  void applyLink(ItemVectorView group) { ARCANE_UNUSED(group); }
};

NUMERICS_END_NAMESPACE
ARCANE_END_NAMESPACE

#endif /* ARCGEOSIM_GEOMETRY_IMPL_GEOMETRYTEMPLATES_H */
