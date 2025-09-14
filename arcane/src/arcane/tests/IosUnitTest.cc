// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IosUnitTest.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Service du test des formats d'entrée/sortie du maillage.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/DomUtils.h"
#include "arcane/core/Directory.h"

#include "arcane/tests/IosUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des Ios
 */
class IosUnitTest
: public ArcaneIosUnitTestObject
{
 public:

  explicit IosUnitTest(const ServiceBuildInfo& sbi);
  ~IosUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  bool _testIosWriterReader(IMesh* mesh, const String& file_extension,
                            const String& service_base_name, Integer index);
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_IOSUNITTEST(IosUnitTest,IosUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IosUnitTest::
IosUnitTest(const ServiceBuildInfo& mb)
: ArcaneIosUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IosUnitTest::
~IosUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IosUnitTest::
_testIosWriterReader(IMesh* this_mesh, const String& file_extension,
                     const String& service_base_name, Integer z)
{
  info() << "\t[_testIosWriterReader] " << file_extension;

  // Otherwise, prepare to do the test
  String this_directory(".");
  ISubDomain* sd = subDomain();
  //	IServiceMng* sm = sd->serviceMng();
	IParallelMng* pm = sd->parallelMng();
	IApplication* app = sd->application();
	IMainFactory* main_factory = app->mainFactory();
	
	ScopedPtrT<IXmlDocumentHolder> xdoc(domutils::createXmlDocument());
	XmlNode dummyXmlNode = xdoc->documentNode();
  Directory write_directory(sd->exportDirectory());
  StringBuilder output_file_name(options()->outputFileName());
  String file_name = output_file_name.toString() + "." + file_extension.lower();
  String full_file_name = write_directory.file(file_name);
  info() << "\t[_testIosWriterReader] " << file_extension << "Mesh(" << full_file_name << ")";

  // Prepare the mesh writer
  String writer_service_name(service_base_name + "MeshWriter");
  auto mesh_writer(ServiceBuilder<IMeshWriter>::createReference(sd, writer_service_name));

  // Now write to file
  info() << "\t[_testIosWriterReader] writeMeshToFile service=" << writer_service_name << "(" << file_name << ")";
  if (mesh_writer->writeMeshToFile(this_mesh, full_file_name)) {
    info() << "\t[_testIosWriterReader] ERROR while " << writer_service_name << "(" << file_name << ")";
    return false;
  }

  // Prepare the mesh reader
  String reader_service_name(service_base_name + "MeshReader");
  auto mesh_reader(ServiceBuilder<IMeshReader>::createReference(sd, reader_service_name));
  IPrimaryMesh* iMesh = main_factory->createMesh(sd, pm->sequentialParallelMng(), reader_service_name + z);
  info() << "\t[_testIosWriterReader] " << reader_service_name << "(" << file_name << ")";

  // Now read to file
  info() << "\t[_testIosWriterReader] " << reader_service_name << "(" << file_name << ")";
  if (mesh_reader->readMeshFromFile(iMesh, dummyXmlNode, full_file_name, this_directory, true) != IMeshReader::RTOk) {
    info() << "\t[_testIosWriterReader] ERROR while " << reader_service_name << "(" << file_name << ")";
    return false;
  }

  // And check for validity
  iMesh->checkValidMesh();

  info() << "\t[_testIosWriterReader] done";
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Ce test va lancer séquentiellement une écriture suivit d'une lecture
 de l'ensemble des maillages lus à l'initialisation.
 [SOD|SIMPLE(1)|SIMPLE(2)] -> wVTU -> rVTU -> wXMF -> rXMF -> wMsh -> rMsh,
 en checkValidMesh-ant ceux lus.
 Un test plus consistant serait à effectuer d'ailleur sur ceux-ci.
*/
void IosUnitTest::
executeTest()
{
	info() << "[IosUnitTest] executeTest";

	String meshName("Mesh");
	IMesh* current_mesh;
	for (Integer z=0; (current_mesh=subDomain()->findMesh(meshName+z, false)) != 0; ++z){
    IPrimaryMesh* pm = current_mesh->toPrimaryMesh();

		info() << "##############################";
		info() << "[IosUnitTest] Working on mesh " << z << " name=" << current_mesh->name() << " pm=" << pm;
		info() << "NodeFamily1=" << pm->nodeFamily();
		info() << "NodeFamily2=" << current_mesh->nodeFamily();

    if (options()->writeVtu() && (!_testIosWriterReader(current_mesh, "vtu", "VtuNew", z)))
      ARCANE_FATAL("Error in >vtu< test");

    if (options()->writeXmf() && (!_testIosWriterReader(current_mesh, "xmf", "Xmf", z)))
      ARCANE_FATAL("Error in >xmf< test");

    if (options()->writeMsh() && (!_testIosWriterReader(current_mesh, "msh", "MshNew", z)))
      ARCANE_FATAL("Error in >msh< test");

    if (options()->writeVtkLegacy() && !_testIosWriterReader(current_mesh, "vtk", "VtkLegacy", z))
      ARCANE_FATAL("Error in >vtk< test");
  }

  // Pour test, affiche les coordonnées des noeuds des 10 premières mailles
  {
    VariableNodeReal3& nodes_coord_var(mesh()->nodesCoordinates());
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      if (cell.localId()>10)
        break;
      info() << "Cell uid=" << ItemPrinter(cell);
      for(Node node : cell.nodes()){
        info() << "Node uid=" << ItemPrinter(node) << " pos=" << nodes_coord_var[node];
      }
    }
  }
	info() << "[IosUnitTest] done";
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IosUnitTest::
initializeTest()
{
  info() << "[IosUnitTest] initializeTest";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
