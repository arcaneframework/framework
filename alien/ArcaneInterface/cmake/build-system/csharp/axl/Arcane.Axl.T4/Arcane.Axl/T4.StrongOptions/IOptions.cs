﻿// ------------------------------------------------------------------------------
//  <autogenerated>
//      This code was generated by a tool.
//      Mono Runtime Version: 4.0.30319.17020
// 
//      Changes to this file may cause incorrect behavior and will be lost if 
//      the code is regenerated.
//  </autogenerated>
// ------------------------------------------------------------------------------

namespace Arcane.Axl {
    using System.Linq;
    using System.Text;
    using System.Collections.Generic;
    using System;
    
    
    public partial class IOptions : IOptionsBase {
        
        public virtual string TransformText() {
            this.GenerationEnvironment = null;
            
            #line 6 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n");
            
            #line default
            #line hidden
            
            #line 7 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
  Action<OptionHandler, String> bodySection = (OptionHandler _xml, String beginLineSpace) => { 
            
            #line default
            #line hidden
            
            #line 8 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 9 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 foreach(var simple in _xml.Simple) {
	string return_type = simple.type.QualifiedName();
	if (simple.IsSingle == false) {
		return_type = return_type.ToArrayType();
	} 
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture(beginLineSpace));
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("virtual ");
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( return_type ));
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(" ");
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( simple.Name.ToFuncName() ));
            
            #line default
            #line hidden
            
            #line 14 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() const = 0;\n");
            
            #line default
            #line hidden
            
            #line 15 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 16 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 17 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 foreach(var enumerator in _xml.Enumeration) {
	string return_type = enumerator.type;
	if (enumerator.IsSingle == false) {
		return_type = return_type.ToArrayType();
	} 
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture(beginLineSpace));
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("virtual ");
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( return_type ));
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(" ");
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( enumerator.Name.ToFuncName() ));
            
            #line default
            #line hidden
            
            #line 22 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() const = 0;\n");
            
            #line default
            #line hidden
            
            #line 23 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 24 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 25 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 foreach(var service in _xml.ServiceInstance) {
	string return_type = service.type + "*";
	if (service.IsSingle == false) {
		return_type = return_type.ToArrayType();
	} 
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture(beginLineSpace));
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("virtual ");
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( return_type ));
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(" ");
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( service.Name.ToFuncName() ));
            
            #line default
            #line hidden
            
            #line 30 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() const = 0;\n");
            
            #line default
            #line hidden
            
            #line 31 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 32 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 33 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 foreach(var complex in _xml.Complex.Where( p => !p.IsRef) ) {
	string return_type;
	if (complex.IsSingle == true)
		return_type = "const IOptions" + complex.type + "&";
	else {
		return_type = "IOptions" + complex.type + "*";
		return_type = return_type.ToArrayType();
	} 
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture(beginLineSpace));
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("virtual ");
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( return_type ));
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(" ");
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( complex.Name.ToFuncName() ));
            
            #line default
            #line hidden
            
            #line 41 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() const = 0;\n");
            
            #line default
            #line hidden
            
            #line 42 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 43 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 }; 
            
            #line default
            #line hidden
            
            #line 44 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n/*---------------------------------------------------------------------------*/\n/*---------------------------------------------------------------------------*/\n// #WARNING#: This file has been generated automatically. Do not edit.\n// Arcane version ");
            
            #line default
            #line hidden
            
            #line 48 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Version ));
            
            #line default
            #line hidden
            
            #line 48 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(" : ");
            
            #line default
            #line hidden
            
            #line 48 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( DateTime.Now ));
            
            #line default
            #line hidden
            
            #line 48 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n/*---------------------------------------------------------------------------*/\n/*---------------------------------------------------------------------------*/\n\n#ifndef ARCANE_IOPTIONS_");
            
            #line default
            #line hidden
            
            #line 52 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Xml.ClassName.ToUpper() ));
            
            #line default
            #line hidden
            
            #line 52 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("_H\n#define ARCANE_IOPTIONS_");
            
            #line default
            #line hidden
            
            #line 53 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Xml.ClassName.ToUpper() ));
            
            #line default
            #line hidden
            
            #line 53 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("_H\n\n/*---------------------------------------------------------------------------*/\n/*---------------------------------------------------------------------------*/\n\n");
            
            #line default
            #line hidden
            
            #line 58 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 if(WithArcane) { 
            
            #line default
            #line hidden
            
            #line 59 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("#include \"arcane/VariableTypes.h\"\n");
            
            #line default
            #line hidden
            
            #line 60 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 61 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n/*---------------------------------------------------------------------------*/\n/*---------------------------------------------------------------------------*/\n\n");
            
            #line default
            #line hidden
            
            #line 65 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 var non_ref_complex = Xml.FlatteningComplex.Where(p => !p.IsRef); 
            
            #line default
            #line hidden
            
            #line 66 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("class IOptions");
            
            #line default
            #line hidden
            
            #line 66 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Xml.ClassName ));
            
            #line default
            #line hidden
            
            #line 66 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n{\npublic:\n\n");
            
            #line default
            #line hidden
            
            #line 70 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // I - Complex Options Definition 
            
            #line default
            #line hidden
            
            #line 71 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 72 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 foreach(var complex in non_ref_complex ) { 
            
            #line default
            #line hidden
            
            #line 73 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("  class IOptions");
            
            #line default
            #line hidden
            
            #line 73 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( complex.type ));
            
            #line default
            #line hidden
            
            #line 73 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n  {\n  public:\n");
            
            #line default
            #line hidden
            
            #line 76 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 77 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // I - Complex Options 
            
            #line default
            #line hidden
            
            #line 78 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 bodySection(complex.Xml, "    "); 
            
            #line default
            #line hidden
            
            #line 79 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("    virtual ~IOptions");
            
            #line default
            #line hidden
            
            #line 79 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( complex.type ));
            
            #line default
            #line hidden
            
            #line 79 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() {}\n  };\n\n");
            
            #line default
            #line hidden
            
            #line 82 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 } 
            
            #line default
            #line hidden
            
            #line 83 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // II - Main service options 
            
            #line default
            #line hidden
            
            #line 84 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 85 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("  virtual ~IOptions");
            
            #line default
            #line hidden
            
            #line 85 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Xml.ClassName ));
            
            #line default
            #line hidden
            
            #line 85 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("() {}\n");
            
            #line default
            #line hidden
            
            #line 86 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 // 
            
            #line default
            #line hidden
            
            #line 87 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
 bodySection(Xml, "  "); 
            
            #line default
            #line hidden
            
            #line 88 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("\n};\n\n/*---------------------------------------------------------------------------*/\n/*---------------------------------------------------------------------------*/\n\n#endif // ARCANE_IOPTIONS_");
            
            #line default
            #line hidden
            
            #line 94 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write(this.ToStringHelper.ToStringWithCulture( Xml.ClassName.ToUpper() ));
            
            #line default
            #line hidden
            
            #line 94 "/work/irlin665_1/gaynor/ALIEN_FINAL_STRONG_OPTIONS/Alien/cmake/build-system/csharp/axl/Arcane.Axl.T4/Arcane.Axl/T4.StrongOptions/IOptions.tt"
            this.Write("_H");
            
            #line default
            #line hidden
            return this.GenerationEnvironment.ToString();
        }
        
        protected virtual void Initialize() {
        }
    }
    
    public class IOptionsBase {
        
        private global::System.Text.StringBuilder builder;
        
        private global::System.Collections.Generic.IDictionary<string, object> session;
        
        private global::System.CodeDom.Compiler.CompilerErrorCollection errors;
        
        private string currentIndent = string.Empty;
        
        private global::System.Collections.Generic.Stack<int> indents;
        
        private ToStringInstanceHelper _toStringHelper = new ToStringInstanceHelper();
        
        public virtual global::System.Collections.Generic.IDictionary<string, object> Session {
            get {
                return this.session;
            }
            set {
                this.session = value;
            }
        }
        
        public global::System.Text.StringBuilder GenerationEnvironment {
            get {
                if ((this.builder == null)) {
                    this.builder = new global::System.Text.StringBuilder();
                }
                return this.builder;
            }
            set {
                this.builder = value;
            }
        }
        
        protected global::System.CodeDom.Compiler.CompilerErrorCollection Errors {
            get {
                if ((this.errors == null)) {
                    this.errors = new global::System.CodeDom.Compiler.CompilerErrorCollection();
                }
                return this.errors;
            }
        }
        
        public string CurrentIndent {
            get {
                return this.currentIndent;
            }
        }
        
        private global::System.Collections.Generic.Stack<int> Indents {
            get {
                if ((this.indents == null)) {
                    this.indents = new global::System.Collections.Generic.Stack<int>();
                }
                return this.indents;
            }
        }
        
        public ToStringInstanceHelper ToStringHelper {
            get {
                return this._toStringHelper;
            }
        }
        
        public void Error(string message) {
            this.Errors.Add(new global::System.CodeDom.Compiler.CompilerError(null, -1, -1, null, message));
        }
        
        public void Warning(string message) {
            global::System.CodeDom.Compiler.CompilerError val = new global::System.CodeDom.Compiler.CompilerError(null, -1, -1, null, message);
            val.IsWarning = true;
            this.Errors.Add(val);
        }
        
        public string PopIndent() {
            if ((this.Indents.Count == 0)) {
                return string.Empty;
            }
            int lastPos = (this.currentIndent.Length - this.Indents.Pop());
            string last = this.currentIndent.Substring(lastPos);
            this.currentIndent = this.currentIndent.Substring(0, lastPos);
            return last;
        }
        
        public void PushIndent(string indent) {
            this.Indents.Push(indent.Length);
            this.currentIndent = (this.currentIndent + indent);
        }
        
        public void ClearIndent() {
            this.currentIndent = string.Empty;
            this.Indents.Clear();
        }
        
        public void Write(string textToAppend) {
            this.GenerationEnvironment.Append(textToAppend);
        }
        
        public void Write(string format, params object[] args) {
            this.GenerationEnvironment.AppendFormat(format, args);
        }
        
        public void WriteLine(string textToAppend) {
            this.GenerationEnvironment.Append(this.currentIndent);
            this.GenerationEnvironment.AppendLine(textToAppend);
        }
        
        public void WriteLine(string format, params object[] args) {
            this.GenerationEnvironment.Append(this.currentIndent);
            this.GenerationEnvironment.AppendFormat(format, args);
            this.GenerationEnvironment.AppendLine();
        }
        
        public class ToStringInstanceHelper {
            
            private global::System.IFormatProvider formatProvider = global::System.Globalization.CultureInfo.InvariantCulture;
            
            public global::System.IFormatProvider FormatProvider {
                get {
                    return this.formatProvider;
                }
                set {
                    if ((this.formatProvider == null)) {
                        throw new global::System.ArgumentNullException("formatProvider");
                    }
                    this.formatProvider = value;
                }
            }
            
            public string ToStringWithCulture(object objectToConvert) {
                if ((objectToConvert == null)) {
                    throw new global::System.ArgumentNullException("objectToConvert");
                }
                global::System.Type type = objectToConvert.GetType();
                global::System.Type iConvertibleType = typeof(global::System.IConvertible);
                if (iConvertibleType.IsAssignableFrom(type)) {
                    return ((global::System.IConvertible)(objectToConvert)).ToString(this.formatProvider);
                }
                global::System.Reflection.MethodInfo methInfo = type.GetMethod("ToString", new global::System.Type[] {
                            iConvertibleType});
                if ((methInfo != null)) {
                    return ((string)(methInfo.Invoke(objectToConvert, new object[] {
                                this.formatProvider})));
                }
                return objectToConvert.ToString();
            }
        }
    }
}
