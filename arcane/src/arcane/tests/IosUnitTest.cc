// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IosUnitTest.cc                                              (C) 2000-2024 */
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

	bool _testIosWriterReader(IMesh* mesh, bool option, String ext, Integer);
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

/*****************************************************************\
 * IosUnitTest
\*****************************************************************/

IosUnitTest::
~IosUnitTest()
{
}


/*****************************************************************\
 * _testIosWriterReader
\*****************************************************************/
bool IosUnitTest::
_testIosWriterReader(IMesh* this_mesh, bool option, String Ext, Integer z)
{
	// If we aren't to be there, just tell it
  if (option != true)
    return false;
	
	info() << "\t[_testIosWriterReader] " << Ext;

	// Otherwise, prepare to do the test
	String this_directory(".");
	ISubDomain* sd = subDomain();
  //	IServiceMng* sm = sd->serviceMng();
	IParallelMng* pm = sd->parallelMng();
	IApplication* app = sd->application();
	IMainFactory* main_factory = app->mainFactory();
	
	ScopedPtrT<IXmlDocumentHolder> xdoc(domutils::createXmlDocument());
	XmlNode dummyXmlNode = xdoc->documentNode();

	StringBuilder outputFileName(options()->outputFileName());
	String file_name=outputFileName.toString();// + "." + Ext.lower();
	String file_name_with_ext=outputFileName.toString() + "." + Ext.lower();
	info() << "\t[_testIosWriterReader] " << Ext << "Mesh(" << file_name_with_ext <<")";
	
	// Prepare the mesh writer
	String writerServiceName(Ext+"NewMeshWriter");
	auto meshWriter(ServiceBuilder<IMeshWriter>::createReference(sd,writerServiceName));
	//FactoryT<IMeshWriter> meshWriter_factory(sd->serviceMng());
	//meshWriter = meshWriter_factory.createInstance(writerServiceName,true);
	//if (!meshWriter.get())
  //throw FatalErrorException(A_FUNCINFO, "Can not create the "+writerServiceName);

	// Now write to file
	info() << "\t[_testIosWriterReader] writeMeshToFile service=" << writerServiceName << "(" << file_name <<")";
	if (meshWriter->writeMeshToFile(this_mesh, file_name)){
	  info() << "\t[_testIosWriterReader] ERROR while " << writerServiceName << "(" << file_name <<")";
	  return false;
	}
	
	// Prepare the mesh reader
	String readerServiceName(Ext+"NewMeshReader");
	auto meshReader(ServiceBuilder<IMeshReader>::createReference(sd,readerServiceName));
	IPrimaryMesh* iMesh = main_factory->createMesh(sd,pm->sequentialParallelMng(), readerServiceName+z);
	info() << "\t[_testIosWriterReader] " << readerServiceName << "(" << file_name <<")";

	// Now read to file
	info() << "\t[_testIosWriterReader] " << readerServiceName << "(" << file_name <<")";
	if (meshReader->readMeshFromFile(iMesh, dummyXmlNode, file_name_with_ext, this_directory, true) != IMeshReader::RTOk){
	  info() << "\t[_testIosWriterReader] ERROR while "<< readerServiceName << "(" << file_name <<")";
	  return false;
	}

	// And check for validity
	iMesh->checkValidMesh();
	
	info() << "\t[_testIosWriterReader] done";
	//TODO: il ne faut pas détruire le maillage directement mais prévoir
  // un appel au sous-domain qui ferait cela.
  //delete iMesh;
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
	
		if ((options()->writeVtu()) && (!_testIosWriterReader(current_mesh, options()->writeVtu(), "Vtu", z)))
			ARCANE_FATAL("Error in >vtu< test");

		if ((options()->writeXmf()) && (!_testIosWriterReader(current_mesh, options()->writeXmf(), "Xmf", z)))
      ARCANE_FATAL("Error in >xmf< test");

		if ((options()->writeMsh()) && (!_testIosWriterReader(current_mesh, options()->writeMsh(), "Msh", z)))
      ARCANE_FATAL("Error in >msh< test");
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
