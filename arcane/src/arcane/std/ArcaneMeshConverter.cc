// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMeshConverter.cc                                      (C) 2000-2020 */
/*                                                                           */
/* Service de partitionnement externe du maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/BasicService.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IDirectExecution.h"

#include "arcane/ServiceBuilder.h"

#include "arcane/std/ArcaneMeshConverter_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de conversion de format du maillage.
 *
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

 private:
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
  bool r = mesh()->utilities()->writeToFile(file_name,options()->writerServiceName());
  if (r)
    ARCANE_FATAL("Error during mesh conversion");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANEMESHCONVERTER(ArcaneMeshConverter,ArcaneMeshConverter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
