// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%ignore Arcane::IMesh::_compactMng;
%ignore Arcane::IMesh::_connectivityPolicy;
%ignore Arcane::IMesh::tiedInterfaces;
%ignore Arcane::IMesh::computeTiedInterfaces(const XmlNode&);
%ignore Arcane::IMesh::serializeCells;
%ignore Arcane::IMesh::meshItemInternalList;


%include arcane/core/IMeshBase.h
%include arcane/core/MeshHandle.h
%include arcane/core/IMesh.h
%include arcane/core/IPrimaryMesh.h
%include arcane/core/IMeshModifier.h
%include arcane/core/MeshPartInfo.h
%include arcane/core/IMeshUtilities.h
%include arcane/core/IItemConnectivityInfo.h
