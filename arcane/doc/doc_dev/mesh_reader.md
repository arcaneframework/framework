# Réalisation d'un lecteur de maillage {#arcanedoc_mesh_reader}

Cette page décrit la manière de réaliser un lecteur de maillage pour Arcane.

Un lecteur de maillage est un service qui implémente l'interface IMeshReader, décrite ci-dessous:

```cpp
class IMeshReader
{
 public:

  virtual bool allowExtension(const String& str) =0;
 
  virtual eReturnType readMeshFromFile(IMesh* mesh,
                                       const XmlNode& mesh_element,
                                       const String& file_name,
                                       const String& dir_name,
                                       bool use_internal_partition) =0;
};
```

La première chose à faire est donc de définir une classe service qui implémente cette interface. Notre classe s'appellera
'SampleMeshReader' et héritera de BasicService. Ce service aura pour nom 'SampleMeshReaderService'

```cpp
#include "arcane/BasicService.h"
#include "arcane/IMeshReader.h"
//! Pour la définition de la macro ARCANE_REGISTER_SUB_DOMAIN_FACTORY
#include "arcane/FactoryService.h"

using namespace Arcane;

class SampleMeshReader
: public BasicService
, public IMeshReader
{
public:
SampleMeshReader(const ServiceBuildInfo& sbi);
public:
virtual bool allowExtension(const String& str);
virtual eReturnType readMeshFromFile(IMesh* mesh,
                                     const XmlNode& mesh_element,
                                     const String& file_name,
																		 const String& dir_name,
																		 bool use_internal_partition);
};

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(SampleMeshReader,IMeshReader,SampleMeshReaderService);
```

La méthode IMeshReader::allowExtension() permet de spécifier l'extension de fichier que notre
lecteur supportera. Par exemple, cette extension est 'vtu' pour les fichiers VTK contenant
des maillages non structurés. Dans notre exemple, nous prendrons l'extension 'msh'. Nous
implémenterons donc la méthode allowExtension() comme suit:

```cpp
bool SampleMeshReader::allowExtension(const String& str)
{
  return str=='msh';
}
```

TODO: a continuer
