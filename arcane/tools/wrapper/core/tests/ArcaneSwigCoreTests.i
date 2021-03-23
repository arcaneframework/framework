// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%module(directors="1") ArcaneCoreTests

%import core/ArcaneSwigCore.i

// Teste que les classes standards de la STL sont bien mappées
%include std_vector.i
%include std_map.i

%{
#include "ArcaneSwigUtils.h"
#include <vector>
#include <map>
  using namespace Arcane;
%}

%template(IntVector) std::vector<int>;
%template(IntMap) std::map<int,int>;
