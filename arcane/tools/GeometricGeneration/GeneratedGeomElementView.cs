﻿// ------------------------------------------------------------------------------
//  <autogenerated>
//      This code was generated by a tool.
//      Mono Runtime Version: 4.0.30319.17020
// 
//      Changes to this file may cause incorrect behavior and will be lost if 
//      the code is regenerated.
//  </autogenerated>
// ------------------------------------------------------------------------------

namespace GeometricGeneration {
    using System;
    
    
    public partial class GeneratedGeomElementView : GeneratedGeomElementViewBase {
        
        public virtual string TransformText() {
            this.GenerationEnvironment = null;
            
            
            this.Write("/*\n * Génération des classes des vues sur les éléments géométriques (GeomElement).\n */\n");
            
            
            
            
            
 foreach( Item item in Connectivity.Items2DAnd3D ) { 
            
            
            
            
            
            this.Write("\n/*!\n * \\ingroup ArcaneGeometric\n * \\brief Vue constante sur les éléments géométriques de type GeomType::");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write(".\n * \n * Il est possible de récupérer une vue de ce type via:\n * - directement depuis une instance de ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ShapeView.\n * - directement depuis une instance de ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("Element\n * - une instance de ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView via ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView::constView()\n * - une instance de GeomShapeView via GeomShapeView::to");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("Element()\n *\n * Pour plus d'informations sur l'usage, se reporter à \\ref arcanedoc_cea_geometric_viewusage\n */\nclass ARCANE_CEA_GEOMETRIC_EXPORT ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView\n: public GeomElementConstViewBase\n{\n public:\n  ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView(ARCANE_RESTRICT const Real3POD* ptr)\n  : GeomElementConstViewBase(ptr){}\n};\n\n/*!\n * \\ingroup ArcaneGeometric\n * \\brief Vue modifiable sur les éléments géométriques de type GeomType::");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write(".\n * \n * Il est possible de récupérer une vue de ce type via:\n * - directement depuis une instance de ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("Element\n * - une instance de ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("Element via ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView::view()\n *\n * Pour plus d'informations sur l'usage, se reporter à \\ref arcanedoc_cea_geometric_viewusage\n */\nclass ARCANE_CEA_GEOMETRIC_EXPORT ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView\n: public GeomElementViewBase\n{\n public:\n  typedef ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView ConstViewType;\n public:\n  ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView(ARCANE_RESTRICT Real3POD* ptr)\n  : GeomElementViewBase(ptr){}\n  //! Initialise la vue avec les coordonnées passées en argument\n  void init(");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.CoordsArgString()));
            
            
            
            
            
            this.Write(")\n  {\n   ");
            
            
            
            
            
 for( int i=0; i<item.NbNode; ++i ) { 
            
            
            
            
            
            this.Write("   m_s[");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(i));
            
            
            
            
            
            this.Write("] = a");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(i));
            
            
            
            
            
            this.Write(";\n   ");
            
            
            
            
            
 } 
            
            
            
            
            
            this.Write("  }\n  //! Opérateur de conversion vers une vue constante\n  operator ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView() const { return ConstViewType(m_s); }\n  //! Vue constante sur l'élément\n  ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView constView() const { return ConstViewType(m_s); }\n};\n\n");
            
            
            
            
            
 // Génère un typedef s'il existe un nom court 
            
            
            
            
            
 if (item.BasicName != item.Name ) { 
            
            
            
            
            
            this.Write("//! Vue sur un élément de type GeomType::");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("\ntypedef ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementView ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.BasicName));
            
            
            
            
            
            this.Write("ElementView;\n//! Vue constante sur un élément de type GeomType::");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("\ntypedef ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.Name));
            
            
            
            
            
            this.Write("ElementConstView ");
            
            
            
            
            
            this.Write(this.ToStringHelper.ToStringWithCulture(item.BasicName));
            
            
            
            
            
            this.Write("ElementConstView;\n");
            
            
            
            
            
 } 
            
            
            
            
            
            this.Write("\n");
            
            
            
            
            
 } 
            
            
            
            return this.GenerationEnvironment.ToString();
        }
        
        public virtual void Initialize() {
        }
    }
    
    public class GeneratedGeomElementViewBase {
        
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
                    if ((value != null)) {
                        this.formatProvider = value;
                    }
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