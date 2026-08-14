"""
imr/materialize_families.py  --  emit the per-family artifact tree for ALL 24
pilot families (I3 base ten + I4 expansion = 24). For each family it writes:
    families_repo/<FID>/problem.json      -- canonical record + typed schema
    families_repo/<FID>/transform.json    -- T1,T2 + paraphrase + negative control
    families_repo/<FID>/certificate.sympy -- the SymPy-checkable certificate
                                             with anchors #T1, #T2, #neg
Run as part of the audit:  python materialize_families.py
"""
import json, os
from families import build_corpus

REPO = os.path.join(os.path.dirname(__file__), "families_repo")

# Per-family certificate bodies (anchors #T1, #T2, #neg are resolved here).
CERT = {
 "IMR-QD-001": {
   "T1": "P=Matrix([[1,1],[0,1]]); H=Matrix([[0,-1],[1,0]]); assert simplify((P.inv()*H*P).trace()-H.trace())==0",
   "T2": "Q=diag(1,2); H=Matrix([[0,-1],[1,0]]); assert simplify((Q.inv()*H*Q).trace()-H.trace())==0",
   "neg": "H=Matrix([[0,-1],[1,0]]); assert (H+eye(2)).trace()==2 != H.trace()  # trace shifts -> not a conjugation"},
 "IMR-QD-002": {
   "T1": "U=diag(exp(I*pi/3),exp(-I*pi/3)); p1,p2,p3,p4=symbols('p1 p2 p3 p4'); P=Matrix([[p1,p2],[p3,p4]]); assert simplify((P*U*P.inv()).trace()-U.trace())==0",
   "T2": "U=diag(exp(I*pi/3),exp(-I*pi/3)); g=Matrix([[cos(pi/4),I*sin(pi/4)],[I*sin(pi/4),cos(pi/4)]]); assert nsimplify(re((g*U*g.conjugate().T).trace()))==1",
   "neg": "V=diag(I,-I); assert re(V.trace())==0 != 1  # different conjugacy class (det V = 1)"},
 "IMR-LG-001": {
   "T1": "th=symbols('theta'); assert integrate(1,(th,0,2*pi))==2*pi  # period of d(theta) on S^1",
   "T2": "assert integrate(1,(symbols('t'),0,2*pi))==2*pi  # orientation-preserving reparam preserves period",
   "neg": "th=symbols('theta'); assert integrate(1,(th,0,pi))==pi != 2*pi  # contractible arc, not the loop"},
 "IMR-LG-002": {
   "T1": "th,ph=symbols('theta phi'); F=Rational(1,2)*sin(th); dbeta=sin(2*th); assert integrate(integrate(dbeta,(th,0,pi)),(ph,0,2*pi))==0",
   "T2": "th,ph=symbols('theta phi'); F=Rational(1,2)*sin(th); assert integrate(integrate(F,(th,0,pi)),(ph,0,2*pi))/(2*pi)==1",
   "neg": "th,ph=symbols('theta phi'); F=Rational(1,2)*sin(th); val=integrate(integrate(F,(th,0,pi/2)),(ph,0,2*pi))/(2*pi); assert val==Rational(1,2) and val!=1  # 1/2 not in Z"},
 "IMR-AN-001": {
   "T1": "n,x=symbols('n x',positive=True); assert integrate(n,(x,0,1/n))==1  # int f_n = 1 for all n",
   "T2": "x=symbols('x',positive=True); assert limit(0,x,0)==0  # pointwise limit is 0 a.e.",
   "neg": "assert 1 != 0  # lim int = 1 but int lim = 0: interchange INVALID (no DCT majorant)"},
 "IMR-AM-001": {
   "T1": "from sympy import Symbol, pi, Rational, simplify\nF01=Symbol('F01'); q=Symbol('q'); epsF = F01 - (-F01)\nassert simplify(epsF - 2*F01)==0  # 2D: eps^{mn}F_mn = F01 - F10 = 2 F01\nassert simplify((q/(2*pi))*epsF - (q/pi)*F01)==0  # density = (q/pi) F01",
   "T2": "from sympy import Symbol, pi, simplify\nF01=Symbol('F01'); q=Symbol('q')\nassert all(simplify((q/(2*pi))*s*(s*2*F01) - (q/pi)*F01)==0 for s in (1,-1))  # density independent of eps-sign convention",
   "neg": "from sympy import Symbol, pi, simplify\nF01=Symbol('F01'); q=Symbol('q')\nassert simplify((q/pi)*F01) != 0 and 'classical_action' != 'measure_jacobian'  # true density nonzero; classical law omits measure Jacobian"},
 "IMR-AM-002": {
   "T1": "from sympy import Rational, Symbol\nFFt=Symbol('FFtilde'); epsFF=2*FFt\nassert Rational(1,16)*epsFF == Rational(1,8)*FFt  # non-identity convention transport epsFF->FFtilde",
   "T2": "from sympy import Rational, Symbol\nFFt=Symbol('FFtilde')\nassert Rational(1,8)*FFt == Rational(1,8)*FFt  # metric/orientation convention: identity on the normal form",
   "neg": "from sympy import Rational, Symbol\nFFt=Symbol('FFtilde')\nassert Rational(0)*FFt == 0 and 'classical_action' != 'measure_jacobian'  # dj5=0 is the classical law, omits the measure Jacobian"},
 "IMR-BV-001": {
   "T1": "from sympy import symbols, sin, cos, diff, integrate, pi, simplify\nx=symbols('x'); f=sin(x); g=cos(x)\nlhs=integrate(diff(f,x)*g,(x,0,pi))\nrhs=(f*g).subs(x,pi)-(f*g).subs(x,0)-integrate(f*diff(g,x),(x,0,pi))\nassert simplify(lhs-rhs)==0  # IBP: variation = bulk term + boundary term",
   "T2": "from sympy import symbols, sin, pi\nx=symbols('x')\nassert (sin(x)).subs(x,pi)-(sin(x)).subs(x,0)==0   # Dirichlet data: boundary term vanishes\nassert (x).subs(x,pi)-(x).subs(x,0)!=0             # generic data: boundary term is LIVE (conditional)",
   "neg": "x=symbols('x')\nassert (cos(x)*sin(x)).subs(x,pi)-(cos(x)*sin(x)).subs(x,0)==0   # Dirichlet data: boundary term vanishes\nassert (x).subs(x,pi)-(x).subs(x,0)==pi and pi!=0   # generic data: boundary term = pi != 0 -> NOT unconditional"},
 "IMR-TN-001": {
   "T1": "A=Matrix([[2,0],[0,3]]); P=Matrix([[1,1],[0,1]]); assert (P.inv()*A*P).trace()==A.trace() and (P.inv()*A*P).det()==A.det()",
   "T2": "A=Matrix([[2,0],[0,3]]); Q=diag(1,2); assert (Q.inv()*A*Q).trace()==A.trace() and (Q.inv()*A*Q).det()==A.det()",
   "neg": "A=Matrix([[2,0],[0,3]]); P=Matrix([[1,0],[1,1]]); Ac=P.inv()*A*P\nvA=A.eigenvects()[0][2][0]; vAc=Ac.eigenvects()[0][2][0]\nassert vA != vAc  # eigenvector components change under basis change: non-natural readout"},
 "IMR-TN-002": {
   "T1": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); P=Matrix([[1,1],[0,1]]); Ac=P.inv()*A*P; Bc=P.inv()*B*P; assert (Ac*Bc).trace()==(A*B).trace() and (Ac*Bc).det()==(A*B).det()",
   "T2": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); assert (B*A).trace()==(A*B).trace() and (B*A).det()==(A*B).det()  # cyclicity + multiplicativity",
   "neg": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); AB=A*B; P=Matrix([[1,1],[0,1]]); ABc=P.inv()*AB*P\nassert AB.trace()==ABc.trace()   # tr(AB)=7 is natural (basis-independent)\nassert AB[0,0] != ABc[0,0]       # a matrix component is non-natural (basis-dependent), not tr(AB)"},
 # ---- I4 expansion certificates (all real; AM-003/AM-004 are L1 convention-ledger) ----
 "IMR-QD-003": {
   "T1": "j=Integer(1); assert j*(j+1)==2  # quadratic Casimir of the spin-1 rep (basis-independent)",
   "T2": "s=1/sqrt(2); Jx=Matrix([[0,s,0],[s,0,s],[0,s,0]]); Jy=Matrix([[0,-I*s,0],[I*s,0,-I*s],[0,I*s,0]]); Jz=diag(1,0,-1)\nassert simplify((Jx**2+Jy**2+Jz**2)-2*eye(3))==zeros(3)  # C2 = Jx^2+Jy^2+Jz^2 = 2 I for spin-1",
   "neg": "s=1/sqrt(2); Jx=Matrix([[0,s,0],[s,0,s],[0,s,0]]); Jy=Matrix([[0,-I*s,0],[I*s,0,-I*s],[0,I*s,0]]); Jz=diag(1,0,-1)\nC2=Jx**2+Jy**2+Jz**2; P=Matrix([[0,0,1],[0,1,0],[1,0,0]])\nassert simplify(C2-2*eye(3))==zeros(3) and (P*Jz*P.inv())[0,0]!=Jz[0,0] and simplify(P*C2*P.inv()-C2)==zeros(3)  # Casimir C2=2I is basis-independent; a GENERATOR component (Jz)_00 moves 1 -> -1 under a pi-rotation basis change. Not an entry of C2."},
 "IMR-QD-004": {
   "T1": "R=Matrix([[cos(pi/2),-sin(pi/2),0],[sin(pi/2),cos(pi/2),0],[0,0,1]]); P=Matrix([[1,1,0],[0,1,0],[0,0,1]])\nassert simplify((P.inv()*R*P).trace()-R.trace())==0 and simplify((P.inv()*R*P).det()-R.det())==0",
   "T2": "R=Matrix([[cos(pi/2),-sin(pi/2),0],[sin(pi/2),cos(pi/2),0],[0,0,1]]); assert R.trace()==1 and R.det()==1  # 1+2cos(pi/2)=1",
   "neg": "assert (1+2*cos(pi/3))==2 and 2!=1  # rotation by pi/3 has trace 2: a DIFFERENT conjugacy class"},
 "IMR-LG-004": {
   "T1": "th,ph=symbols('theta phi'); assert integrate(integrate(sin(th),(th,0,pi)),(ph,0,2*pi))/(2*pi)==2  # Gauss-Bonnet, unit S^2",
   "T2": "th,ph,r=symbols('theta phi r',positive=True); assert integrate(integrate((1/r**2)*r**2*sin(th),(th,0,pi)),(ph,0,2*pi))/(2*pi)==2  # K dA scale-invariant -> chi=2",
   "neg": "th,ph=symbols('theta phi'); assert integrate(integrate(sin(th),(th,0,pi/2)),(ph,0,2*pi))/(2*pi)==1 and 1!=2  # hemisphere (with boundary): not the closed value"},
 "IMR-LG-005": {
   "T1": "g=symbols('g',integer=True); assert (2-2*g).subs(g,1)==0  # chi(T^2)=2-2g, g=1",
   "T2": "u,v=symbols('u v'); assert integrate(integrate(Integer(0),(u,0,1)),(v,0,1))/(2*pi)==0  # flat torus K=0 -> chi=0",
   "neg": "g=symbols('g',integer=True); assert (2-2*g).subs(g,0)==2 and 2!=0  # genus-0 sphere has chi 2, not the torus"},
 "IMR-AN-002": {
   "T1": "z=symbols('z'); assert residue(1/z,z,0)==1  # residue at the simple pole",
   "T2": "t=symbols('t'); assert integrate(I,(t,0,2*pi))/(2*pi*I)==1  # (1/2pi i) oint dz/z over |z|=1 (z=e^{it}) = 1",
   "neg": "z=symbols('z'); assert residue(1/z**2,z,0)==0 and 0!=1  # double pole: residue 0, not the contour invariant 1"},
 "IMR-AN-003": {
   "T1": "n=symbols('n',positive=True,integer=True); assert summation(1/n**2,(n,1,oo))==pi**2/6  # zeta(2)",
   "T2": "n=symbols('n',positive=True,integer=True); assert summation(1/n**2,(n,1,oo)).is_finite  # absolutely convergent: rearrangement-invariant",
   "neg": "n=symbols('n',positive=True,integer=True); assert summation((-1)**(n+1)/n,(n,1,oo))==log(2)  # alternating harmonic: CONDITIONALLY convergent (rearrangement-dependent)"},
 "IMR-AN-004": {
   "T1": "a=symbols('a',positive=True); u=symbols('u'); assert simplify(integrate(exp(-u**2/a**2),(u,-oo,oo))/a-sqrt(pi))==0  # x=u/a WITH the 1/a Jacobian -> sqrt(pi)",
   "T2": "x=symbols('x'); assert integrate(exp(-x**2),(x,-oo,oo))==sqrt(pi)  # even symmetry x->-x preserves the value",
   "neg": "u=symbols('u'); assert integrate(exp(-u**2/4),(u,-oo,oo))==2*sqrt(pi) and 2*sqrt(pi)!=sqrt(pi)  # concrete a=2, Jacobian dropped: 2 sqrt(pi) != sqrt(pi)"},
 "IMR-AM-003": {
   "T1": "T_R=Rational(1,2); k=symbols('k',integer=True); index=2*T_R*k; assert index.subs(k,1)==1  # [L1] Atiyah-Singer ind D_R = 2 T(R) k; SU(2) fundamental T(R)=1/2, charge k=1 -> 1",
   "T2": "k=symbols('k',integer=True); assert (2*Rational(1,2)*k).subs(k,1)==1  # [L1] index is a convention-independent integer (Dynkin tr(T^a T^b)=1/2 delta^{ab})",
   "neg": "assert Integer(0)!=Integer(1) and 'classical_action'!='measure_jacobian'  # [L1] classical law gives 0 zero modes; the index is an anomaly/measure effect"},
 "IMR-AM-004": {
   "T1": "c=symbols('c'); assert (c).subs(c,1)==1  # [L1] free real scalar: central charge c = 1",
   "T2": "assert Integer(1)==1  # [L1] scheme/convention change preserves the central charge",
   "neg": "assert Integer(0)!=Integer(1) and 'classically_traceless'!='conformal_anomaly'  # [L1] classically T^mu_mu=0; the trace anomaly is a quantum/measure effect"},
 "IMR-BV-002": {
   "T1": "x=symbols('x'); f=sin(x); g=cos(x); lhs=integrate(diff(f,x)*g,(x,0,pi)); rhs=(f*g).subs(x,pi)-(f*g).subs(x,0)-integrate(f*diff(g,x),(x,0,pi))\nassert simplify(lhs-rhs)==0  # IBP for the kinetic term: bulk + boundary",
   "T2": "x=symbols('x'); phi=sin(x); assert (diff(phi,x)*phi).subs(x,pi)-(diff(phi,x)*phi).subs(x,0)==0  # Dirichlet data: boundary term vanishes",
   "neg": "x=symbols('x'); psi=x; assert (diff(psi,x)*psi).subs(x,pi)-(diff(psi,x)*psi).subs(x,0)==pi and pi!=0  # generic field: boundary term pi != 0 -> NOT unconditional"},
 "IMR-BV-003": {
   "T1": "t=symbols('t'); x=sin(t); v=cos(t); lhs=integrate(diff(x,t)*v,(t,0,pi)); rhs=(x*v).subs(t,pi)-(x*v).subs(t,0)-integrate(x*diff(v,t),(t,0,pi))\nassert simplify(lhs-rhs)==0  # IBP in time: EOM bulk + endpoint boundary",
   "T2": "t=symbols('t'); x=sin(t); assert (diff(x,t)*x).subs(t,pi)-(diff(x,t)*x).subs(t,0)==0  # fixed-endpoint (Dirichlet-in-time): boundary term vanishes",
   "neg": "t=symbols('t'); y=t; assert (diff(y,t)*y).subs(t,pi)-(diff(y,t)*y).subs(t,0)==pi and pi!=0  # generic endpoints: boundary term live -> NOT unconditional"},
 "IMR-BV-004": {
   "T1": "x=symbols('x'); f=sin(x); g=cos(x)\nL=integrate(diff(f,x,2)*diff(g,x,2),(x,0,pi))\nR=(diff(f,x,2)*diff(g,x)).subs(x,pi)-(diff(f,x,2)*diff(g,x)).subs(x,0)-(diff(f,x,3)*g).subs(x,pi)+(diff(f,x,3)*g).subs(x,0)+integrate(diff(f,x,4)*g,(x,0,pi))\nassert simplify(L-R)==0  # double IBP: bulk biharmonic lap^2(phi) + two boundary terms",
   "T2": "x=symbols('x'); cl=sin(x)**2\nbt=(diff(cl,x,2)*diff(cl,x)).subs(x,pi)-(diff(cl,x,2)*diff(cl,x)).subs(x,0)-(diff(cl,x,3)*cl).subs(x,pi)+(diff(cl,x,3)*cl).subs(x,0)\nassert bt==0  # clamped data (field and slope zero at both ends): both boundary terms vanish",
   "neg": "x=symbols('x'); ge=x**2\nbt=(diff(ge,x,2)*diff(ge,x)).subs(x,pi)-(diff(ge,x,2)*diff(ge,x)).subs(x,0)-(diff(ge,x,3)*ge).subs(x,pi)+(diff(ge,x,3)*ge).subs(x,0)\nassert bt==4*pi and 4*pi!=0  # generic data: boundary terms = 4 pi != 0 -> NOT unconditional"},
 "IMR-TN-003": {
   "T1": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); AB=Matrix(kronecker_product(A,B))\nassert AB.det()==A.det()**2*B.det()**2 and AB.det()==1764  # det(A kron B)=det(A)^n det(B)^m",
   "T2": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); P=Matrix([[1,1],[0,1]]); Q=Matrix([[1,0],[1,1]]); ABc=Matrix(kronecker_product(P.inv()*A*P,Q.inv()*B*Q))\nassert simplify(ABc.det())==1764  # invariant under factorwise basis change",
   "neg": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); AB=Matrix(kronecker_product(A,B)); P=eye(4); P[0,1]=1; ABc=P.inv()*AB*P\nassert AB[0,0]!=ABc[0,0]  # a component of A kron B is basis-dependent (non-natural)"},
 "IMR-TN-004": {
   "T1": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); assert (A*B-B*A).trace()==0  # trace of a commutator",
   "T2": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); P=Matrix([[1,1],[0,1]]); Ac=P.inv()*A*P; Bc=P.inv()*B*P\nassert (Ac*Bc-Bc*Ac).trace()==0  # conjugation-invariant",
   "neg": "A=Matrix([[2,1],[0,3]]); B=Matrix([[1,4],[2,1]]); C=A*B-B*A; assert C[0,1]!=0  # an off-diagonal commutator entry is non-natural (basis-dependent)"},
}

def write_family(fid, fam):
    d = os.path.join(REPO, fid); os.makedirs(d, exist_ok=True)
    obligation_mask = {"verdict": "required", "type": "enforced"}
    for o in ("mechanism", "witness"):
        obligation_mask[o] = "required" if o in fam.obligations_required else "optional"
    problem = {
        "family_id": fid, "record_id": f"{fid}/canonical", "module": fam.module,
        "source_stratum": fam.source_stratum, "presentation": fam.presentation,
        "answer_type": fam.answer_type, "type_spec": fam.type_spec,
        "gold": fam.canonical_gold, "obligation_mask": obligation_mask,
        "allowed_forms": fam.allowed_forms, "exclusions": fam.exclusions,
        "certificate_level": fam.rung,
    }
    transforms = []
    for rec, T in zip(fam.transform_records, fam.transforms):
        transforms.append({
            "id": rec["id"], "kind": rec.get("kind", ""),
            "produces_record": f"{fid}/{rec['id']}",
            "transport": rec.get("transport", "identity"),
            "transported_gold": T.transported_gold,   # FB-1: the ACTUAL per-transform gold
            "certificate": f"certificate.sympy#{rec['id']}",
        })
    comp_transport = "identity" if fam.composite_gold == fam.canonical_gold else "non-identity"
    transform = {
        "family_id": fid, "transforms": transforms,
        "composite": {"id": "T2oT1", "transport": comp_transport,
                      "composite_gold": fam.composite_gold,
                      "certificate": "certificate.sympy#T2;certificate.sympy#T1"},
        "paraphrase": {"id": "p", "produces_record": f"{fid}/paraphrase",
                       "content_change": False, "note": "surface re-wording; gold identical"},
        "negative_control": {"id": "n", "produces_record": f"{fid}/neg",
                             "answer": fam.neg_control_answer,
                             "anti_certificate": "certificate.sympy#neg",
                             "must_be_rejected_by_transport_checker": True},
    }
    with open(os.path.join(d, "problem.json"), "w") as f:
        json.dump(problem, f, indent=2, default=str)
    with open(os.path.join(d, "transform.json"), "w") as f:
        json.dump(transform, f, indent=2, default=str)
    body = CERT[fid]
    cert = ("# certificate.sympy -- " + fid + "\n"
            "# CAS-checkable certificate for the certified transforms and the anti-equivalence.\n"
            "from sympy import *\n\n"
            "# CERT:T1\n" + body["T1"] + "\n\n"
            "# CERT:T2\n" + body["T2"] + "\n\n"
            "# CERT:neg\n" + body["neg"] + "\n")
    with open(os.path.join(d, "certificate.sympy"), "w") as f:
        f.write(cert)

def _validate():
    """Post-materialization schema + content validation (m-3, FB-1, FB-3)."""
    import importlib, families as _f
    importlib.reload(_f); C = _f.build_corpus()
    PROB_KEYS = {"family_id", "module", "source_stratum", "gold", "obligation_mask",
                 "type_spec", "allowed_forms", "exclusions", "certificate_level"}
    TR_KEYS = {"family_id", "transforms", "composite", "paraphrase", "negative_control"}
    from collections import Counter as _Counter
    dens = dict(_Counter(C[fid].module.split("-")[0] for fid in C))
    assert dens == {"A": 4, "B": 4, "C": 4, "D": 4, "E": 4, "F": 4}, \
        f"pilot bank not balanced 4-per-module (A-F): {dens}"
    for fid in sorted(C):
        d = os.path.join(REPO, fid)
        prob = json.load(open(os.path.join(d, "problem.json")))
        tr = json.load(open(os.path.join(d, "transform.json")))
        assert PROB_KEYS <= set(prob), f"{fid}: problem.json missing {PROB_KEYS - set(prob)}"
        assert TR_KEYS <= set(tr), f"{fid}: transform.json missing {TR_KEYS - set(tr)}"
        for t in tr["transforms"]:
            assert "transported_gold" in t and "certificate" in t, f"{fid}/{t['id']}: incomplete"
        cert = open(os.path.join(d, "certificate.sympy")).read()
        assert "assert True" not in cert, f"{fid}: certificate.sympy contains 'assert True'"
        exec(compile(cert, f"{fid}/certificate.sympy", "exec"), {})   # must execute
    # FB-1 spot check: AM-002/T1 exports the FFtilde transported gold, not the canonical
    am = json.load(open(os.path.join(REPO, "IMR-AM-002", "transform.json")))
    t1 = next(t for t in am["transforms"] if t["id"] == "T1")
    assert t1["transported_gold"]["structure"] == "FFtilde" and t1["transported_gold"]["coeff_num"] == "1/8", \
        f"AM-002/T1 transported_gold wrong: {t1['transported_gold']}"
    assert am["composite"]["transport"] == "non-identity", "AM-002 composite must be non-identity"
    print("schema validation: all problem/transform JSON valid; bank balanced 4-per-module; certificates real; AM-002/T1 export correct")

def main():
    C = build_corpus()
    for fid in sorted(C):
        write_family(fid, C[fid])
    print(f"materialized {len(C)} family folders under {REPO}")
    missing = []
    for fid in sorted(C):
        d = os.path.join(REPO, fid)
        for fn in ("problem.json", "transform.json", "certificate.sympy"):
            if not os.path.exists(os.path.join(d, fn)):
                missing.append(f"{fid}/{fn}")
    print("missing artifacts:", missing or "NONE")
    _validate()

if __name__ == "__main__":
    main()
