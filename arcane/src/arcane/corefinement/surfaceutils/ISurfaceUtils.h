// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCGEOSIM_SURFACEUTILS_ISURFACEUTILS_H
#define ARCGEOSIM_SURFACEUTILS_ISURFACEUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/Real3.h>
#include <arcane/core/Item.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Numerics
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISurface;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISurfaceUtils
{
 public:

  struct FaceFaceContact
  {
    FaceFaceContact() {}
    FaceFaceContact(const FaceFaceContact& c)
    : faceA(c.faceA)
    , faceB(c.faceB)
    , centerA(c.centerA)
    , centerB(c.centerB)
    , normalA(c.normalA)
    , normalB(c.normalB)
    {}
    FaceFaceContact(const Face& fA, const Face& fB)
    : faceA(fA)
    , faceB(fB)
    {}
    Face faceA, faceB;
    Real3 centerA, centerB;
    Real3 normalA, normalB;
  };
  typedef UniqueArray<FaceFaceContact> FaceFaceContactList;

 public:

  //! Constructor
  ISurfaceUtils() {}

  //! Destructor
  virtual ~ISurfaceUtils() {};

  //! Initialization
  virtual void init() = 0;

  //! Creation of a new surface
  virtual ISurface* createSurface() = 0;

  //! Sets the faces of a surface
  virtual void setFaceToSurface(ISurface* surface, FaceGroup face_group) = 0;

  //! compute for each face of surface1 the nearest face of surface2
  virtual void computeSurfaceContact(ISurface* surface1,
                                     ISurface* surface2,
                                     FaceFaceContactList& contact) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Numerics

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
