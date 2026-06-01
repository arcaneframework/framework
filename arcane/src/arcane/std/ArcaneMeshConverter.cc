// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMeshConverter.cc                                      (C) 2000-2020 */
/*                                                                           */
/* External mesh partitioning service.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IDirectExecution.h"

#include "arcane/core/ServiceBuilder.h"

#include "arcane/std/ArcaneMeshConverter_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh format conversion service.
 */
class ArcaneMeshConverter
: public ArcaneArcaneMeshConverterObject
{
 public:
 public:

  explicit ArcaneMeshConverter(const ServiceBuildInfo& cb);

 public:

  void build() override {}
  void execute() override;
  void setParallelMng(IParallelMng*) override {}
  bool isActive() const override { return true; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMeshConverter::
ArcaneMeshConverter(const ServiceBuildInfo& sb)
: ArcaneArcaneMeshConverterObject(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMeshConverter::
execute()
{
  String file_name = options()->fileName();
  if (file_name.empty())
    ARCANE_FATAL("Invalid null option file-name");
  bool r = mesh()->utilities()->writeToFile(file_name, options()->writerServiceName());
  if (r)
    ARCANE_FATAL("Error during mesh conversion");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANEMESHCONVERTER(ArcaneMeshConverter, ArcaneMeshConverter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
