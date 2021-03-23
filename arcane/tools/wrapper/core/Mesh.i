// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%ignore Arcane::IMesh::_compactMng;
%ignore Arcane::IMesh::_connectivityPolicy;
%ignore Arcane::IMesh::tiedInterfaces;
%ignore Arcane::IMesh::computeTiedInterfaces(const XmlNode&);
%ignore Arcane::IMesh::serializeCells;
%ignore Arcane::IMesh::meshItemInternalList;

%include arcane/MeshHandle.h
%include arcane/IMesh.h
%include arcane/IPrimaryMesh.h
%include arcane/IMeshModifier.h
%include arcane/MeshPartInfo.h
%include arcane/IMeshUtilities.h
%include arcane/IItemConnectivityInfo.h
