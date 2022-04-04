// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshGeneratorService.cc                                     (C) 2000-2020 */
/*                                                                           */
/* Service de génération de maillage.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/IMeshReader.h"
#include "arcane/ISubDomain.h"
#include "arcane/XmlNode.h"
#include "arcane/Service.h"
#include "arcane/IParallelMng.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/MeshVariable.h"
#include "arcane/ItemPrinter.h"
#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"

#include "arcane/std/SodMeshGenerator.h"
#include "arcane/std/SimpleMeshGenerator.h"
#include "arcane/std/CartesianMeshGenerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de génération de maillages
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
  if (!meshclass.null()){
    gen_node = mesh_node.childWithNameAttr(meshgen_ustr,meshclass);
    info() << "Using ARCANE_MESHGENERATOR";
  }else{
    gen_node = mesh_node.child(meshgen_ustr);
  }
  if (gen_node.null()) return RTIrrelevant;
  XmlNode node;
  // Recherche du 'cartesian' mesh generator
  if (node.null()){
    node = gen_node.child("cartesian");
    if (!node.null())
      generator = new CartesianMeshGenerator(mesh);
  }
  // Recherche du 'sod' mesh generator
  if (node.null()){
    node = gen_node.child("sod");
    if (!node.null())
      generator = new SodMeshGenerator(mesh, node.attr("zyx").valueAsBoolean());
  }
  // Recherche du 'simple' mesh generator
  if (node.null()){
    node = gen_node.child("simple");
    if (!node.null())
      generator = new SimpleMeshGenerator(mesh);
  }
  if (!generator.get()){
    warning() << "Type de générateur de maillage inconnu.";
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

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(MeshGeneratorService,IMeshReader,MeshGenerator);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
