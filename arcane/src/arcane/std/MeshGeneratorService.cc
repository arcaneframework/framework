// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshGeneratorService.cc                                     (C) 2000-2020 */
/*                                                                           */
/* Mesh generation service.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/Service.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"

#include "arcane/std/SodMeshGenerator.h"
#include "arcane/std/SimpleMeshGenerator.h"
#include "arcane/std/CartesianMeshGenerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh generation service
 */
class MeshGeneratorService
: public AbstractService
, public IMeshReader
{
 public:

  MeshGeneratorService(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

 public:

  virtual bool allowExtension(const String& str)
  {
    return str.null();
  }
  virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,
                                       const XmlNode& mesh_node,
                                       const String& file_name,
                                       const String& dirname,
                                       bool use_internal_partition);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshGeneratorService::
MeshGeneratorService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MeshGeneratorService::
readMeshFromFile(IPrimaryMesh* mesh,
                 const XmlNode& mesh_node,
                 const String& filename,
                 const String& dirname,
                 bool use_internal_partition)
{
  ARCANE_UNUSED(filename);
  ARCANE_UNUSED(dirname);
  ARCANE_UNUSED(use_internal_partition);

  String meshclass = platform::getEnvironmentVariable("ARCANE_MESHGENERATOR");
  String meshgen_ustr("meshgenerator");
  XmlNode gen_node;
  ScopedPtrT<IMeshGenerator> generator;

  //msg->warning() << "USING MESHGENERATOR !!\n";
  if (!meshclass.null()) {
    gen_node = mesh_node.childWithNameAttr(meshgen_ustr, meshclass);
    info() << "Using ARCANE_MESHGENERATOR";
  }
  else {
    gen_node = mesh_node.child(meshgen_ustr);
  }
  if (gen_node.null())
    return RTIrrelevant;
  XmlNode node;
  // Searching for the 'cartesian' mesh generator
  if (node.null()) {
    node = gen_node.child("cartesian");
    if (!node.null())
      generator = new CartesianMeshGenerator(mesh);
  }
  // Searching for the 'sod' mesh generator
  if (node.null()) {
    node = gen_node.child("sod");
    if (!node.null())
      generator = new SodMeshGenerator(mesh, node.attr("zyx").valueAsBoolean());
  }
  // Searching for the 'simple' mesh generator
  if (node.null()) {
    node = gen_node.child("simple");
    if (!node.null())
      generator = new SimpleMeshGenerator(mesh);
  }
  if (!generator.get()) {
    warning() << "Unknown mesh generator type.";
    return RTIrrelevant;
  }
  if (generator->readOptions(node))
    return RTError;
  if (generator->generateMesh())
    return RTError;
  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(MeshGeneratorService, IMeshReader, MeshGenerator);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
