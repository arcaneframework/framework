# Implementing a mesh reader {#arcanedoc_mesh_reader}

This page describes how to implement a mesh reader for Arcane.

A mesh reader is a service that implements the IMeshReader interface, described
below:

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

The first thing to do is therefore define a service class that implements this
interface. Our class will be called 'SampleMeshReader' and will inherit from
BasicService. This service will be named 'SampleMeshReaderService'

```cpp
#include "arcane/BasicService.h"
#include "arcane/IMeshReader.h"
//! For the definition of the ARCANE_REGISTER_SUB_DOMAIN_FACTORY macro
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

The IMeshReader::allowExtension() method allows specifying the file extension
that our reader will support. For example, this extension is 'vtu' for VTK files
containing unstructured meshes. In our example, we will use the 'msh' extension.
We will therefore implement the allowExtension() method as follows:

```cpp
bool SampleMeshReader::allowExtension(const String& str)
{
  return str=='msh';
}
```

TODO: to continue
