// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshWriter.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh writing service.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHWRITER_H
#define ARCANE_CORE_IMESHWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface of a mesh writing service.
 */
class IMeshWriter
{
 public:

  virtual ~IMeshWriter() {} //<! Releases resources

 public:

  //! Constructs the instance
  virtual void build() =0;

 public:

  /*!
   * \brief Writes a mesh to a file.
   *
   * The mesh file path must be writable and
   * the directory must already exist.
   *
   * \param mesh mesh to save
   * \param file_name name of the mesh file.
   *
   * \retval true in case of error
   * \retval false if everything is ok.
   */
  virtual bool writeMeshToFile(IMesh* mesh,const String& file_name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
