import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Plus, Trash2, Play, Layout, Maximize2, ZoomIn, ZoomOut, 
  Download, Activity, BarChart2, Save, FileJson, Hash, 
  Layers, Info, Package, Hammer, RotateCcw, ArrowDown, Move, MousePointer2,
  Anchor, Settings
} from 'lucide-react';

// --- 核心数学与力学计算模块 ---
const SM_Engine = {
  zeros: (r, c) => Array.from({ length: r }, () => Array(c).fill(0)),
  
  // Matrix Multiplication
  multiply: (A, B) => {
    const rA = A.length, cA = A[0].length, rB = B.length, cB = B[0].length;
    if (cA !== rB) throw new Error("Matrix dimensions mismatch");
    const C = Array.from({ length: rA }, () => Array(cB).fill(0));
    for (let i = 0; i < rA; i++)
      for (let j = 0; j < cB; j++)
        for (let k = 0; k < cA; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  },

  // Matrix Transpose
  transpose: (A) => {
    const r = A.length, c = A[0].length;
    const T = Array.from({ length: c }, () => Array(r).fill(0));
    for(let i=0; i<r; i++) for(let j=0; j<c; j++) T[j][i] = A[i][j];
    return T;
  },

  // Small Matrix Inversion (Gaussian elimination)
  inv: (A) => {
    const n = A.length;
    const X = Array.from({ length: n }, (_, i) => {
      const row = Array(n).fill(0);
      row[i] = 1;
      return row;
    });
    // Create augmented matrix [A | I]
    const M = A.map((row, i) => [...row, ...X[i]]);
    
    for (let i = 0; i < n; i++) {
      let max = i;
      for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[max][i])) max = j;
      [M[i], M[max]] = [M[max], M[i]];
      const pivot = M[i][i];
      if (Math.abs(pivot) < 1e-12) throw new Error("Singular matrix in inversion");
      for (let k = i; k < 2 * n; k++) M[i][k] /= pivot;
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const f = M[j][i];
          for (let k = i; k < 2 * n; k++) M[j][k] -= f * M[i][k];
        }
      }
    }
    return M.map(row => row.slice(n));
  },

  // Static Condensation
  // K: 6x6, F: 6x1 (FEF), releasedIndices: [2, 5] (indices of DOFs to release in local system)
  // Returns { K_cond, F_cond, T_cond } where T_cond helps recover released DOFs
  condense: (K, F, releasedIndices) => {
    if (!releasedIndices || releasedIndices.length === 0) return { K, F, releasedIndices: [] };

    const n = 6;
    const all = [0,1,2,3,4,5];
    const r = releasedIndices;
    const k = all.filter(i => !r.includes(i));
    
    if (k.length === 0) return { K: SM_Engine.zeros(6,6), F: Array(6).fill(0), releasedIndices };

    // Extract submatrices
    const Kkk = k.map(i => k.map(j => K[i][j]));
    const Kkr = k.map(i => r.map(j => K[i][j]));
    const Krk = r.map(i => k.map(j => K[i][j]));
    const Krr = r.map(i => r.map(j => K[i][j]));
    
    const Fk = k.map(i => F[i]);
    const Fr = r.map(i => F[i]); // These are fixed end forces at the hinge. Usually we want internal moment = 0.
    // The FEF calculated by calcFEF assumes fixed ends.
    // If released, the boundary condition is Force_r = 0 (externally). 
    // But internally, the element has FEF_r.
    // The equation is: Krr * ur + Krk * uk + Fr = 0 (Total force at released DOF is 0)
    // So ur = -Krr_inv * (Krk * uk + Fr)
    
    const Krr_inv = SM_Engine.inv(Krr);
    
    // T_mat = Krr_inv * Krk
    const T_mat = SM_Engine.multiply(Krr_inv, Krk);
    
    // K_cond = Kkk - Kkr * T_mat
    const Kkr_T = SM_Engine.multiply(Kkr, T_mat);
    const K_cond_small = Kkk.map((row, i) => row.map((val, j) => val - Kkr_T[i][j]));
    
    // F_cond_adj = Krr_inv * Fr
    const F_cond_adj_vec = Fr.map(v => [v]); // Column vector
    const F_adj = SM_Engine.multiply(Krr_inv, F_cond_adj_vec).flat();
    
    // F_cond = Fk - Kkr * F_adj
    const Kkr_F = SM_Engine.multiply(Kkr, F_adj.map(v=>[v])).flat();
    const F_cond_small = Fk.map((v, i) => v - Kkr_F[i]);
    
    // Reassemble into 6x6
    const K_new = SM_Engine.zeros(6, 6);
    const F_new = Array(6).fill(0);
    
    for(let i=0; i<k.length; i++) {
        F_new[k[i]] = F_cond_small[i];
        for(let j=0; j<k.length; j++) {
            K_new[k[i]][k[j]] = K_cond_small[i][j];
        }
    }
    
    return { K: K_new, F: F_new, Krr_inv, Krk, Fr, releasedIndices, kIndices: k };
  },

  // 线性方程组求解 (带全主元的高斯消元)
  solve: (K, P) => {
    const n = P.length;
    const A = K.map((row, i) => [...row, P[i]]);
    for (let i = 0; i < n; i++) {
      let max = i;
      for (let j = i + 1; j < n; j++) if (Math.abs(A[j][i]) > Math.abs(A[max][i])) max = j;
      [A[i], A[max]] = [A[max], A[i]];
      
      // Singularity Check: If pivot is too small, matrix is likely singular (unstable structure)
      if (Math.abs(A[i][i]) < 1e-12) {
        throw new Error("Structure is Unstable or Mechanism detected. Please check supports and connectivity.");
      }

      for (let j = i + 1; j < n; j++) {
        const f = A[j][i] / A[i][i];
        for (let k = i; k <= n; k++) A[j][k] -= f * A[i][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = 0;
      for (let j = i + 1; j < n; j++) s += A[i][j] * x[j];
      x[i] = (A[i][n] - s) / A[i][i];
    }
    return x;
  },

  /**
   * 计算单元固端内力 (FEF) - 支持多态荷载系统
   * f = [u1, v1, m1, u2, v2, m2] (局部坐标系)
   * 
   * 1. Nodal Loads (F_x, F_y, M) - 处理在全局矩阵组装阶段
   * 2. Element Loads (FEF Conversion):
   *    - Point Load (P at a)
   *    - Point Moment (M at a)
   *    - Uniform Load (q)
   *    - Trapezoidal Load (q1 to q2)
   */
  calcFEF: (L, load) => {
    let f = [0, 0, 0, 0, 0, 0];
    const a = load.pos !== undefined ? load.pos : L / 2;
    const b = L - a;

    switch(load.type) {
      case 'uniform': {
        const q = load.val1 || 0;
        // 垂直于杆轴的均布力 q
        const V = (q * L) / 2;
        const M = (q * L * L) / 12;
        f = [0, V, M, 0, V, -M];
        break;
      }
      case 'point_force': {
        const P = load.val1 || 0;
        if (load.dir === 'axial') {
          // 轴向力 P
          f = [P * b / L, 0, 0, P * a / L, 0, 0];
        } else {
          // 垂直力 P
          const M1 = (P * a * b * b) / (L * L);
          const M2 = -(P * a * a * b) / (L * L);
          const V1 = (P * b * b * (3 * a + b)) / (L ** 3);
          const V2 = (P * a * a * (a + 3 * b)) / (L ** 3);
          f = [0, V1, M1, 0, V2, M2];
        }
        break;
      }
      case 'point_moment': {
        const M0 = load.val1 || 0;
        // 集中力矩 M
        const M1 = (M0 * b * (2 * a - b)) / (L * L);
        const M2 = (M0 * a * (2 * b - a)) / (L * L);
        const V1 = -(6 * M0 * a * b) / (L ** 3);
        const V2 = (6 * M0 * a * b) / (L ** 3);
        f = [0, V1, M1, 0, V2, M2];
        break;
      }
      case 'trapezoidal': {
        const q1 = load.val1 || 0;
        const q2 = load.val2 || 0;
        // 梯形荷载 q1 -> q2
        const M1 = (L * L / 60) * (3 * q1 + 2 * q2);
        const M2 = -(L * L / 60) * (2 * q1 + 3 * q2);
        const V1 = (L / 20) * (7 * q1 + 3 * q2);
        const V2 = (L / 20) * (3 * q1 + 7 * q2);
        f = [0, V1, M1, 0, V2, M2];
        break;
      }
      default: break;
    }
    return f;
  },

  resolveLoad: (l, ux, uy) => {
    let fx = 0, fy = 0, m = 0;
    if (l.type === 'point_moment') m = l.val1 || 0;
    else if (l.type === 'point_force') {
      const v = l.val1 || 0;
      if (l.dir === 'axial') { fx = v * ux; fy = v * uy; }
      else { fx = v * (-uy); fy = v * ux; }
    }
    return { fx, fy, m };
  },

  getInterpolatedForces: (L, startForces, loads, segments = 40) => {
    // startForces: [axial, shear, moment] at x=0
    // Sign Convention:
    // x: 0 -> L
    // Shear V: Clockwise Positive (Load Down = dV/dx = -q)
    // Moment M: Sagging Positive (Tension at Bottom, dM/dx = V)
    // Loads: Positive Value = DOWN (Gravity Direction)

    const points = [];
    let [currN, currV, currM] = startForces;
    const dx = L / segments;

    // Initial point
    points.push({ x: 0, shear: currV, moment: currM, axial: currN });

    for (let i = 1; i <= segments; i++) {
      const xPrev = (i - 1) * dx;
      const xCurr = i * dx;
      const xMid = (xPrev + xCurr) / 2;

      // 1. Calculate Distributed Load Intensity q(x) at mid-step
      let q = 0;
      loads.forEach(l => {
        if (l.type === 'uniform') {
           q += (l.val1 || 0);
        } else if (l.type === 'trapezoidal') {
           const q1 = l.val1 || 0;
           const q2 = l.val2 || 0;
           // Linear interp
           q += q1 + (q2 - q1) * (xMid / L);
        }
      });

      // 2. Update Shear (dV = -q * dx)
      // Note: We use q at xMid for better accuracy
      const dV_dist = -q * dx;
      
      // Check for Point Forces in this segment (xPrev < pos <= xCurr)
      let dV_point = 0;
      loads.forEach(l => {
        if (l.type === 'point_force') {
          if (l.pos > xPrev && l.pos <= xCurr + 1e-9) {
             // Point Load P (Positive = Down).
             // Downward load causes Shear to drop?
             // Left: Up reaction. Right: Down load.
             // V goes from Positive to Negative. So V drops.
             // V_new = V_old - P
             dV_point -= (l.val1 || 0);
          }
        }
      });

      const nextV = currV + dV_dist + dV_point;

      // 3. Update Moment (dM = V * dx)
      // Use Average Shear for trapezoidal integration: M_new = M_old + (V_old + V_new)/2 * dx
      // However, the V used for moment should be the V *before* the point load jump at the exact point?
      // Simplified: Use average of V_prev and (currV + dV_dist). Point load jump happens "at" the point.
      // Ideally: Integrate V(x).
      // V(x) in segment is linear (due to q).
      // M_change_dist = (currV + (currV + dV_dist)) / 2 * dx;
      // Point force effect on moment is handled by the change in V for *subsequent* steps, 
      // but if the point force is IN this step, does it cause Moment jump? No. Point Force causes Shear Jump (Slope change in M).
      
      const vAvg = (currV + (currV + dV_dist)) / 2;
      let nextM = currM + vAvg * dx;

      // Check for Point Moments (M jump)
      loads.forEach(l => {
        if (l.type === 'point_moment') {
           if (l.pos > xPrev && l.pos <= xCurr + 1e-9) {
             // Moment M0 (Positive = CCW).
             // CCW Moment causes Drop in Internal Moment Diagram.
             // M_new = M_old - M0
             nextM -= (l.val1 || 0);
           }
        }
      });

      // Update state
      currV = nextV;
      currM = nextM;
      
      points.push({ x: xCurr / L, shear: currV, moment: currM, axial: currN });
    }

    return points;
  },

  preprocess: (nodes, elements, elementLoads, nodalLoads) => {
    let newNodes = [...nodes.map(n => ({...n}))];
    let newElements = [...elements.map(e => ({...e}))];
    let newElementLoads = [...elementLoads.map(l => ({...l}))];
    let newNodalLoads = [...nodalLoads.map(l => ({...l}))];

    let maxNodeId = newNodes.reduce((max, n) => Math.max(max, n.id), 0);
    let maxElementId = newElements.reduce((max, e) => Math.max(max, e.id), 0);
    let maxLoadId = newElementLoads.reduce((max, l) => Math.max(max, l.id), 0);

    const originalElIds = elements.map(e => e.id);

    for (const elId of originalElIds) {
      const elIndex = newElements.findIndex(e => e.id === elId);
      if (elIndex === -1) continue;
      const el = newElements[elIndex];
      const n1 = newNodes.find(n => n.id === el.n1);
      const n2 = newNodes.find(n => n.id === el.n2);
      if (!n1 || !n2) continue;

      const dx = n2.x - n1.x, dy = n2.y - n1.y, L = Math.sqrt(dx*dx + dy*dy);
      
      // 筛选单元内部的集中荷载
      const allLoadsOnThisEl = elementLoads.filter(l => l.elementId === elId);
      const loadsOnEl = allLoadsOnThisEl.filter(l => (l.type === 'point_force' || l.type === 'point_moment') && l.pos > 1e-4 && l.pos < L - 1e-4);
      
      if (loadsOnEl.length === 0) continue;

      // 按位置分组
      const splitPoints = [];
      loadsOnEl.forEach(l => {
         const existing = splitPoints.find(sp => Math.abs(sp.pos - l.pos) < 1e-4);
         if (existing) existing.loads.push(l);
         else splitPoints.push({ pos: l.pos, loads: [l] });
      });
      splitPoints.sort((a, b) => a.pos - b.pos);
      
      // 移除原单元及其荷载
      newElements.splice(elIndex, 1);
      newElementLoads = newElementLoads.filter(l => l.elementId !== elId);
      
      const loadsAtEnds = allLoadsOnThisEl.filter(l => (l.type === 'point_force' || l.type === 'point_moment') && (l.pos <= 1e-4 || l.pos >= L - 1e-4));
      const distributedLoads = allLoadsOnThisEl.filter(l => l.type !== 'point_force' && l.type !== 'point_moment');
      
      const ux = dx / L, uy = dy / L;
      
      // 处理起点荷载
      loadsAtEnds.filter(l => l.pos <= 1e-4).forEach(l => {
         const { fx, fy, m } = SM_Engine.resolveLoad(l, ux, uy);
         newNodalLoads.push({ nodeId: el.n1, fx, fy, m });
      });

      let prevNodeId = el.n1;
      let startPos = 0;
      const segments = [...splitPoints, { pos: L, loads: [], isEnd: true }];

      for (const seg of segments) {
         let nextNodeId;
         if (seg.isEnd) {
             nextNodeId = el.n2;
             loadsAtEnds.filter(l => l.pos >= L - 1e-4).forEach(l => {
                 const { fx, fy, m } = SM_Engine.resolveLoad(l, ux, uy);
                 newNodalLoads.push({ nodeId: nextNodeId, fx, fy, m });
             });
         } else {
             maxNodeId++;
             nextNodeId = maxNodeId;
             newNodes.push({ id: nextNodeId, x: n1.x + ux * seg.pos, y: n1.y + uy * seg.pos });
             seg.loads.forEach(l => {
                 const { fx, fy, m } = SM_Engine.resolveLoad(l, ux, uy);
                 newNodalLoads.push({ nodeId: nextNodeId, fx, fy, m });
             });
         }

         maxElementId++;
         const newElId = maxElementId;
         
         // Inherit releases logic
         const segReleases = [0, 0];
         if (el.releases) {
            if (Math.abs(startPos - 0) < 1e-6) segReleases[0] = el.releases[0]; // Start of original element
            if (seg.isEnd) segReleases[1] = el.releases[1]; // End of original element
         }

         newElements.push({ id: newElId, n1: prevNodeId, n2: nextNodeId, sectionId: el.sectionId, releases: segReleases });

         distributedLoads.forEach(ol => {
             maxLoadId++;
             const newL = { ...ol, id: maxLoadId, elementId: newElId };
             if (ol.type === 'trapezoidal') {
                 const q1 = ol.val1 || 0, q2 = ol.val2 || 0;
                 const x1 = startPos, x2 = seg.pos;
                 newL.val1 = q1 + (q2 - q1) * (x1 / L);
                 newL.val2 = q1 + (q2 - q1) * (x2 / L);
             }
             newElementLoads.push(newL);
         });

         prevNodeId = nextNodeId;
         startPos = seg.pos;
      }
    }
    return { nodes: newNodes, elements: newElements, elementLoads: newElementLoads, nodalLoads: newNodalLoads };
  }
};

const App = () => {
  // --- 模型状态 ---
  const [sections, setSections] = useState([
    { id: 1, name: "混凝土梁截面", E: 30000000, A: 0.12, I: 0.004 },
    { id: 2, name: "钢管截面", E: 206000000, A: 0.015, I: 0.0002 }
  ]);
  const [nodes, setNodes] = useState([
    { id: 1, x: 0, y: 0 }, { id: 2, x: 6, y: 0 }, { id: 3, x: 6, y: -4 }, { id: 4, x: 0, y: -4 }
  ]);
  const [elements, setElements] = useState([
    { id: 1, n1: 1, n2: 2, sectionId: 1 }, { id: 2, n1: 2, n2: 3, sectionId: 1 }, { id: 3, n1: 4, n2: 1, sectionId: 2 }
  ]);
  const [supports, setSupports] = useState([
    { nodeId: 3, type: 'fixed' }, { nodeId: 4, type: 'pinned' }
  ]);
  const [nodalLoads, setNodalLoads] = useState([{ id: 1, nodeId: 2, fx: 0, fy: -15, m: 10 }]);
  const [elementLoads, setElementLoads] = useState([
    { id: 1, elementId: 1, type: 'trapezoidal', val1: 5, val2: 20 },
    { id: 2, elementId: 2, type: 'point_moment', val1: 20, pos: 2 },
    { id: 3, elementId: 2, type: 'point_force', val1: 50, pos: 3, dir: 'vertical' }
  ]);
  
  // --- 视图交互状态 ---
  const [viewMode, setViewMode] = useState('model');
  const [activeTab, setActiveTab] = useState('loads'); 
  const [results, setResults] = useState(null);
  const [transform, setTransform] = useState({ x: 350, y: 350, k: 45 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(null);
  const [showDataCenter, setShowDataCenter] = useState(false);
  const [importText, setImportText] = useState('');
  const canvasRef = useRef(null);

  // --- 数据交互 ---
  const exportData = () => {
    const data = { nodes, elements, sections, nodalLoads, elementLoads, supports };
    setImportText(JSON.stringify(data, null, 2));
    setShowDataCenter(true);
  };

  const importData = () => {
    try {
      const d = JSON.parse(importText);
      if (d.sections) setSections(d.sections);
      if (d.nodes) setNodes(d.nodes);
      if (d.elements) setElements(d.elements);
      if (d.nodalLoads) setNodalLoads(d.nodalLoads);
      if (d.elementLoads) setElementLoads(d.elementLoads);
      if (d.supports) setSupports(d.supports);
      setShowDataCenter(false);
      setResults(null);
    } catch (e) { alert("导入失败：JSON 数据无效"); }
  };

  // --- 交互逻辑 ---
  const handleWheel = (e) => {
    e.preventDefault();
    const factor = Math.pow(1.1, -e.deltaY / 150);
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    setTransform(prev => {
      const newK = Math.max(1, Math.min(prev.k * factor, 1000));
      const actualFactor = newK / prev.k;
      return { k: newK, x: mouseX - (mouseX - prev.x) * actualFactor, y: mouseY - (mouseY - prev.y) * actualFactor };
    });
  };

  const handleMouseDown = (e) => { if (e.button === 1 || e.button === 2) { setIsPanning(true); setLastMouse({ x: e.clientX, y: e.clientY }); } };
  
  const handleMouseMove = (e) => { 
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isPanning) {
        setTransform(prev => ({ ...prev, x: prev.x + (e.clientX - lastMouse.x), y: prev.y + (e.clientY - lastMouse.y) }));
        setLastMouse({ x: e.clientX, y: e.clientY });
        return;
    }
    setLastMouse({ x: e.clientX, y: e.clientY });

    // --- Hover Detection Logic (Smart LOD) ---
    const toCanvas = (wx, wy) => ({ x: transform.x + wx * transform.k, y: transform.y - wy * transform.k });
    const renderElements = (results && results.model) ? results.model.elements : elements;
    const renderNodes = (results && results.model) ? results.model.nodes : nodes;
    
    let found = null;
    
    // 1. Check Nodes for Hover (Priority over elements)
    for (const n of renderNodes) {
        const p = toCanvas(n.x, n.y);
        const dist = Math.hypot(mouseX - p.x, mouseY - p.y);
        if (dist < 10) { 
            found = { type: 'node', id: n.id, x: p.x, y: p.y };
            break;
        }
    }

    // 2. Check Elements for Hover (if no node found)
    if (!found) {
        for (const el of renderElements) {
           const n1 = renderNodes.find(n => n.id === el.n1);
           const n2 = renderNodes.find(n => n.id === el.n2);
           if (!n1 || !n2) continue;
           const p1 = toCanvas(n1.x, n1.y);
           const p2 = toCanvas(n2.x, n2.y);
           
           // Distance from point to line segment
           const l2 = (p1.x-p2.x)**2 + (p1.y-p2.y)**2;
           let t = 0;
           if (l2 !== 0) {
               t = ((mouseX - p1.x) * (p2.x - p1.x) + (mouseY - p1.y) * (p2.y - p1.y)) / l2;
               t = Math.max(0, Math.min(1, t));
           }
           const projX = p1.x + t * (p2.x - p1.x);
           const projY = p1.y + t * (p2.y - p1.y);
           const dist = Math.hypot(mouseX - projX, mouseY - projY);
           
           // Threshold 10px
           if (dist < 10) {
               found = { type: 'element', id: el.id, t, x: projX, y: projY, realX: n1.x + t*(n2.x-n1.x) };
               break; 
           }
        }
    }
    setHovered(found);
  };
  const handleMouseUp = () => setIsPanning(false);

  const autoFit = () => {
    if (nodes.length === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(n => { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y); });
    
    // Add padding (margin) around the structure
    const padding = 100;
    const w = rect.width - padding * 2;
    const h = rect.height - padding * 2;
    
    const k = Math.min(w / (maxX - minX || 1), h / (maxY - minY || 1));
    setTransform({ k, x: rect.width / 2 - ((minX + maxX) / 2) * k, y: rect.height / 2 + ((minY + maxY) / 2) * k });
  };

  // --- 计算引擎 ---
  const solve = () => {
    try {
      const { nodes: pNodes, elements: pElements, elementLoads: pElLoads, nodalLoads: pNodeLoads } = 
        SM_Engine.preprocess(nodes, elements, elementLoads, nodalLoads);

      // --- 1. DOF Mapping & Support Rotation ---
      const nodeAngles = {}; 
      supports.forEach(s => { if (s.angle) nodeAngles[s.nodeId] = s.angle * Math.PI / 180; });

      const dofMap = {};
      let eqCount = 0;
      
      pNodes.forEach(n => {
         const sup = supports.find(s => s.nodeId === n.id);
         const constraints = [0, 0, 0];
         if (sup) {
             if (sup.type === 'fixed') constraints.fill(1);
             else if (sup.type === 'pinned') { constraints[0]=1; constraints[1]=1; }
             else if (sup.type === 'rollerY') constraints[1]=1; // Y constrained
             else if (sup.type === 'rollerX') constraints[0]=1; // X constrained
         }
         dofMap[n.id] = constraints.map(c => c ? -1 : eqCount++);
      });

      // --- 2. Initialize Reduced System ---
      const K_reduced = SM_Engine.zeros(eqCount, eqCount);
      const P_reduced = Array(eqCount).fill(0);
      const elResults = [];

      // --- 3. Element Assembly ---
      pElements.forEach(el => {
        const n1 = pNodes.find(n => n.id === el.n1), n2 = pNodes.find(n => n.id === el.n2);
        const sec = sections.find(s => s.id === el.sectionId);
        if (!n1 || !n2 || !sec) return;

        const dx = n2.x - n1.x, dy = n2.y - n1.y, L = Math.sqrt(dx*dx + dy*dy);
        if (L < 1e-9) return;
        const c = dx/L, s = dy/L, { E, A, I } = sec;
        
        // Local Stiffness
        const EA_L = (E * A) / L, EI12_L3 = (12 * E * I) / (L ** 3), EI6_L2 = (6 * E * I) / (L ** 2), EI4_L = (4 * E * I) / L, EI2_L = (2 * E * I) / L;
        const kl = SM_Engine.zeros(6, 6);
        kl[0][0]=kl[3][3]=EA_L; kl[0][3]=kl[3][0]=-EA_L;
        kl[1][1]=kl[4][4]=EI12_L3; kl[1][4]=kl[4][1]=-EI12_L3;
        kl[1][2]=kl[2][1]=kl[1][5]=kl[5][1]=EI6_L2; kl[4][2]=kl[2][4]=kl[4][5]=kl[5][4]=-EI6_L2;
        kl[2][2]=kl[5][5]=EI4_L; kl[2][5]=kl[5][2]=EI2_L;

        // FEF
        const loads = pElLoads.filter(l => l.elementId === el.id);
        const fef_loc = new Array(6).fill(0);
        loads.forEach(l => { const f = SM_Engine.calcFEF(L, l); f.forEach((v, i) => fef_loc[i] += v); });

        // Condensation
        const relIndices = [];
        if (el.releases && el.releases[0]) relIndices.push(2);
        if (el.releases && el.releases[1]) relIndices.push(5);
        
        const condensed = SM_Engine.condense(kl, fef_loc, relIndices);
        const k_eff = condensed.K;
        const fef_eff = condensed.F;

        // Transformation Matrices
        const T_el = [[c,s,0,0,0,0], [-s,c,0,0,0,0], [0,0,1,0,0,0], [0,0,0,c,s,0], [0,0,0,-s,c,0], [0,0,0,0,0,1]];
        
        const ang1 = nodeAngles[el.n1] || 0, ang2 = nodeAngles[el.n2] || 0;
        const c1 = Math.cos(ang1), s1 = Math.sin(ang1), c2 = Math.cos(ang2), s2 = Math.sin(ang2);
        const R_nodes = [
           [c1, -s1, 0, 0, 0, 0], [s1, c1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
           [0, 0, 0, c2, -s2, 0], [0, 0, 0, s2, c2, 0], [0, 0, 0, 0, 0, 1]
        ];
        
        const T_super = SM_Engine.multiply(T_el, R_nodes);
        const T_super_T = SM_Engine.transpose(T_super);
        
        const k_glob = SM_Engine.multiply(SM_Engine.multiply(T_super_T, k_eff), T_super);
        const fef_glob = Array(6).fill(0);
        for(let i=0; i<6; i++) for(let j=0; j<6; j++) fef_glob[i] += T_super_T[i][j] * fef_eff[j];

        // Assembly
        const indices = [...dofMap[el.n1], ...dofMap[el.n2]];
        
        for(let i=0; i<6; i++) {
          if(indices[i] !== -1) P_reduced[indices[i]] -= fef_glob[i];
          for(let j=0; j<6; j++) {
            if(indices[i] !== -1 && indices[j] !== -1) K_reduced[indices[i]][indices[j]] += k_glob[i][j];
          }
        }
        elResults.push({ id: el.id, k_eff, fef_eff, T_super, condensed, indices });
      });

      // --- 4. Nodal Loads ---
      pNodeLoads.forEach(l => {
        const ang = nodeAngles[l.nodeId] || 0;
        const c = Math.cos(ang), s = Math.sin(ang);
        const fx_rot = l.fx * c + l.fy * s;
        const fy_rot = l.fx * (-s) + l.fy * c;
        const idx = dofMap[l.nodeId];
        if(idx) {
            if(idx[0]!==-1) P_reduced[idx[0]] += fx_rot;
            if(idx[1]!==-1) P_reduced[idx[1]] += fy_rot;
            if(idx[2]!==-1) P_reduced[idx[2]] += l.m || 0;
        }
      });

      // --- 5. Solve ---
      const U_reduced = SM_Engine.solve(K_reduced, P_reduced);

      // --- 6. Post-Process ---
      // Reconstruct Global Displacements (XY System) for Rendering
      const U_xy = new Array(pNodes.length * 3).fill(0);
      pNodes.forEach((n, i) => {
          const idx = dofMap[n.id];
          const ang = nodeAngles[n.id] || 0;
          const u_rot = idx[0]===-1 ? 0 : U_reduced[idx[0]];
          const v_rot = idx[1]===-1 ? 0 : U_reduced[idx[1]];
          const th_rot = idx[2]===-1 ? 0 : U_reduced[idx[2]];
          
          const c = Math.cos(ang), s = Math.sin(ang);
          U_xy[i*3] = u_rot * c - v_rot * s;
          U_xy[i*3+1] = u_rot * s + v_rot * c;
          U_xy[i*3+2] = th_rot;
      });

      // Element Forces
      const forces = elResults.map(er => {
         const u_node_rot = er.indices.map(eq => eq === -1 ? 0 : U_reduced[eq]);
         
         // Geometric Local Displacements
         const u_loc_geo = Array(6).fill(0);
         for(let i=0; i<6; i++) for(let j=0; j<6; j++) u_loc_geo[i] += er.T_super[i][j] * u_node_rot[j];
         
         const f_loc = Array(6).fill(0);
         for(let i=0; i<6; i++) {
             for(let j=0; j<6; j++) f_loc[i] += er.k_eff[i][j] * u_loc_geo[j];
             f_loc[i] += er.fef_eff[i];
         }
         return { id: er.id, moment: [-f_loc[2], f_loc[5]], shear: [f_loc[1], -f_loc[4]], axial: [-f_loc[0], f_loc[3]] };
      });

      setResults({ displacements: U_xy, forces, model: { nodes: pNodes, elements: pElements, elementLoads: pElLoads, nodalLoads: pNodeLoads }, timestamp: Date.now() });
    } catch(e) { 
      console.error(e); 
      alert(e.message || "计算错误。"); 
    }
  };

  // --- 辅助函数：绘制力向量 ---
  // Returns: { tail: {x,y}, tip: {x,y}, angle: radians }
  const getForceVector = (targetX, targetY, fx, fy, scale) => {
    // fx, fy: Force components in Global Coordinates
    // Positive fx: Right. Positive fy: Up.
    // Canvas Y is Down. So Canvas Fy = -fy.
    const canvasFx = fx;
    const canvasFy = -fy; // Invert Y for canvas

    // If we want the arrow tip to be AT the target point (Pushing)
    // Tail = Target - Vector * Scale
    const len = Math.hypot(canvasFx, canvasFy) * scale;
    if (len < 1e-6) return null;

    // Direction vector
    const dirX = canvasFx / (Math.hypot(canvasFx, canvasFy) || 1);
    const dirY = canvasFy / (Math.hypot(canvasFx, canvasFy) || 1);
    
    // Default: Arrow Tip at Target (Pushing)
    // If Force is "Pulling" (like Tension), usually drawn from node outwards.
    // But for "Loads", usually drawn pointing TO the structure.
    // User Requirement: 
    // F > 0 (Up/Right) -> Arrow points Up/Right.
    // F < 0 (Down/Left) -> Arrow points Down/Left.
    // If we draw Tip at Target.
    // Fx > 0 (Right). CanvasFx > 0. DirX > 0.
    // Tail = Target - Dir * Len. Tail is Left of Target. Arrow points Right. Correct.
    // Fy > 0 (Up). CanvasFy < 0. DirY < 0 (Up).
    // Tail = Target - Dir * Len = Target - (Negative) = Target + Positive (Below).
    // Tail is Below Target. Arrow points Up. Correct.
    // Fy < 0 (Down). CanvasFy > 0. DirY > 0 (Down).
    // Tail = Target - Dir * Len = Target - Positive (Above).
    // Tail is Above Target. Arrow points Down. Correct.
    
    return {
       tip: { x: targetX, y: targetY },
       tail: { x: targetX - dirX * len, y: targetY - dirY * len }
    };
  };

  const drawForceArrow = (ctx, tail, tip, color = '#dc2626') => {
      ctx.strokeStyle = color; 
      ctx.fillStyle = color; 
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tail.x, tail.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();

      // Arrow head
      const headLen = 6;
      const angle = Math.atan2(tip.y - tail.y, tip.x - tail.x);
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - headLen * Math.cos(angle - Math.PI/6), tip.y - headLen * Math.sin(angle - Math.PI/6));
      ctx.lineTo(tip.x - headLen * Math.cos(angle + Math.PI/6), tip.y - headLen * Math.sin(angle + Math.PI/6));
      ctx.closePath();
      ctx.fill();
  };

  // --- 绘图 ---
  useEffect(() => {
    // 初始自动适应
    setTimeout(autoFit, 50);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const toCanvas = (wx, wy) => ({ x: transform.x + wx * transform.k, y: transform.y - wy * transform.k });

    const isResultView = results && ['moment', 'shear', 'axial'].includes(viewMode);
    const renderNodes = (isResultView && results.model) ? results.model.nodes : nodes;
    const renderElements = (isResultView && results.model) ? results.model.elements : elements;
    const renderElementLoads = (isResultView && results.model) ? results.model.elementLoads : elementLoads;
    const renderNodalLoads = (isResultView && results.model) ? results.model.nodalLoads : nodalLoads;

    // --- Smart Label System ---
    const occupiedRects = []; // {x, y, w, h}
    const labelList = []; // { text, x, y, color, priority: 0-10, type: 'text'|'box' }

    const addLabel = (text, x, y, color = '#000', priority = 5, align = 'center') => {
        labelList.push({ text, x, y, color, priority, align });
    };

    const drawSmartLabel = (l) => {
        ctx.font = `bold 12px sans-serif`;
        const metrics = ctx.measureText(l.text);
        const padding = 4;
        const width = metrics.width + padding * 2;
        const height = 16 + padding * 2;
        
        let lx = l.x;
        if (l.align === 'center') lx -= width / 2;
        else if (l.align === 'right') lx -= width;
        let ly = l.y - height / 2; // Default centered vertically

        // Collision Resolution (Simple offsets)
        const candidates = [
            { x: lx, y: ly }, // Original
            { x: lx, y: ly - height }, // Top
            { x: lx, y: ly + height }, // Bottom
            { x: lx + width/2, y: ly }, // Right
            { x: lx - width/2, y: ly } // Left
        ];

        let bestPos = null;
        for (const pos of candidates) {
            const r = { x: pos.x, y: pos.y, w: width, h: height };
            let collision = false;
            // Check canvas bounds
            if (r.x < 0 || r.y < 0 || r.x + r.w > w || r.y + r.h > h) collision = true;
            
            // Check other labels
            if (!collision) {
                for (const occ of occupiedRects) {
                    if (r.x < occ.x + occ.w && r.x + r.w > occ.x &&
                        r.y < occ.y + occ.h && r.y + r.h > occ.y) {
                        collision = true;
                        break;
                    }
                }
            }
            if (!collision) {
                bestPos = pos;
                break;
            }
        }

        // If high priority, draw anyway even if collision (fallback to original)
        if (!bestPos && l.priority > 8) bestPos = candidates[0];
        
        if (bestPos) {
            // Draw background halo for readability
            ctx.globalAlpha = 0.8;
            ctx.fillStyle = '#fff';
            ctx.fillRect(bestPos.x, bestPos.y, width, height);
            ctx.globalAlpha = 1.0;
            
            ctx.fillStyle = l.color;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(l.text, bestPos.x + padding, bestPos.y + padding + 1);
            
            occupiedRects.push({ x: bestPos.x, y: bestPos.y, w: width, h: height });
        }
    };

    // --- Visual Hierarchy Settings ---
    // Scale Factors
    let lengths = [];
    renderElements.forEach(el => {
      const n1 = renderNodes.find(n => n.id === el.n1), n2 = renderNodes.find(n => n.id === el.n2);
      if (n1 && n2) {
          const p1 = toCanvas(n1.x, n1.y), p2 = toCanvas(n2.x, n2.y);
          lengths.push(Math.hypot(p2.x - p1.x, p2.y - p1.y));
      }
    });
    const avgLenPx = lengths.length ? lengths.reduce((a,b)=>a+b,0)/lengths.length : Math.min(w,h)/3;
    const loadScalePxPerKN = Math.min(40, Math.max(10, avgLenPx / 80)); // Clamped Arrow Size
    const distLoadScalePxPerKNm = Math.min(30, Math.max(8, avgLenPx / 100));
    
    let diagScalePxPerUnit = 0.5;
    if (isResultView) {
      const vals = results.forces.flatMap(f => {
        const v = f[viewMode];
        return Array.isArray(v) ? v : [];
      });
      const maxAbs = vals.length ? Math.max(...vals.map(v => Math.abs(v))) : 1;
      diagScalePxPerUnit = Math.max(0.1, Math.min(3.0, (avgLenPx / (maxAbs * 8))));
    }

    // --- Layer 1: Grid ---
    ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1;
    const step = 1;
    for(let i = -100; i <= 100; i+=step) {
      const p = toCanvas(i, 0); ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, h); ctx.stroke();
      const q = toCanvas(0, i); ctx.beginPath(); ctx.moveTo(0, q.y); ctx.lineTo(w, q.y); ctx.stroke();
    }

    // --- Layer 2: Structure (Beams) ---
    renderElements.forEach(el => {
      const n1 = renderNodes.find(n => n.id === el.n1), n2 = renderNodes.find(n => n.id === el.n2);
      if (!n1 || !n2) return;
      const p1 = toCanvas(n1.x, n1.y), p2 = toCanvas(n2.x, n2.y);
      
      // Structure Line
      ctx.beginPath(); 
      const isHovered = hovered && hovered.type === 'element' && hovered.id === el.id;
      ctx.strokeStyle = isHovered ? '#6366f1' : '#334155'; 
      ctx.lineWidth = isHovered ? 4 : 3; 
      ctx.lineCap = 'round';
      ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

      // Draw Hinges
      if (el.releases) {
          const dx = p2.x - p1.x, dy = p2.y - p1.y, L = Math.hypot(dx, dy);
          if (L > 16) { // Only draw if long enough
              const ux = dx/L, uy = dy/L;
              ctx.fillStyle = '#fff'; 
              ctx.strokeStyle = '#334155';
              ctx.lineWidth = 1.5;
              
              if (el.releases[0]) {
                  const hx = p1.x + ux * 8; 
                  const hy = p1.y + uy * 8;
                  ctx.beginPath(); ctx.arc(hx, hy, 2.5, 0, 7); ctx.fill(); ctx.stroke();
              }
              if (el.releases[1]) {
                  const hx = p2.x - ux * 8;
                  const hy = p2.y - uy * 8;
                  ctx.beginPath(); ctx.arc(hx, hy, 2.5, 0, 7); ctx.fill(); ctx.stroke();
              }
          }
      }
    });

    // --- Layer 3: Diagrams (M/V/N) ---
    if (isResultView) {
        renderElements.forEach(el => {
            const n1 = renderNodes.find(n => n.id === el.n1), n2 = renderNodes.find(n => n.id === el.n2);
            if (!n1 || !n2) return;
            const p1 = toCanvas(n1.x, n1.y), p2 = toCanvas(n2.x, n2.y);
            const force = results.forces.find(f => f.id === el.id);
            if (!force) return;

            const dx = p2.x - p1.x, dy = p2.y - p1.y;
            const len = Math.hypot(dx, dy);
            // Unit vector along the element
            const ux = dx / len;
            const uy = dy / len;

            // Normal Vector (Local +y)
            // In Structural Mechanics: Local +y is rotated 90 deg Counter-Clockwise from Local x.
            // In Math (Y up): (1,0) -> (0,1).
            // In Canvas (Y down):
            // (1,0) [Right] -> (0, -1) [Up].
            // (0,1) [Down] -> (1, 0) [Right].
            // Formula for 90 deg CCW in Canvas Space:
            // x' = y
            // y' = -x
            // Let's verify: (1,0) -> (0, -1). Correct. (0,1) -> (1, 0). Correct.
            // So Normal Vector (nx, ny) is (uy, -ux).
            const nx = uy;
            const ny = -ux;

            // Color & Style
            let colorHex = '#8b5cf6';
            if (viewMode === 'shear') colorHex = '#10b981';
            else if (viewMode === 'axial') colorHex = '#3b82f6'; // Default Blue for Tension

            ctx.lineWidth = 2;

            // Data Preparation
            const startM = force.moment[0]; 
            const startV = force.shear[0];
            const startN = force.axial[0];
            const elLoads = renderElementLoads.filter(l => l.elementId === el.id);
            const realL = Math.hypot(n2.x - n1.x, n2.y - n1.y);
            const points = SM_Engine.getInterpolatedForces(realL, [startN, startV, startM], elLoads, 40);

            // Draw Polygon
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            
            let maxVal = -Infinity, minVal = Infinity;
            let maxPos = null, minPos = null;
            let centroidX = 0, centroidY = 0;
            let areaSum = 0;

            // Helper to get draw offset vector
            const getDrawOffset = (val, type) => {
                let scale = diagScalePxPerUnit;
                if (type === 'moment') {
                    // Moment Rule: Draw on Tension Side.
                    // Val > 0 (Sagging/Bottom Tension) -> Draw Bottom (Opposite to Normal).
                    // Val < 0 (Hogging/Top Tension) -> Draw Top (Along Normal).
                    // So we always multiply by -Val.
                    // If Val=10, Offset = -10 * Normal (Down). Correct.
                    // If Val=-10, Offset = 10 * Normal (Up). Correct.
                    return -val * scale;
                } else if (type === 'shear') {
                    // Shear Rule: Positive Top (Along Normal).
                    return val * scale;
                } else if (type === 'axial') {
                    // Axial Rule: Positive (Tension) Top (Along Normal).
                    return val * scale;
                }
                return 0;
            };

            points.forEach((pt, i) => {
                let val = 0;
                if (viewMode === 'moment') val = pt.moment; // Raw value
                else if (viewMode === 'shear') val = pt.shear;
                else if (viewMode === 'axial') val = pt.axial;

                // Track Max/Min
                if (val > maxVal) { maxVal = val; maxPos = pt; }
                if (val < minVal) { minVal = val; minPos = pt; }

                const offset = getDrawOffset(val, viewMode);
                const ptx = p1.x + (dx * pt.x);
                const pty = p1.y + (dy * pt.x);
                
                // Final vertex position
                const vx = ptx + nx * offset;
                const vy = pty + ny * offset;
                
                ctx.lineTo(vx, vy);

                // Simple Centroid Accumulation
                centroidX += vx;
                centroidY += vy;
                areaSum++;
            });

            ctx.lineTo(p2.x, p2.y);
            
            // Fill Logic (Axial Special Color)
            if (viewMode === 'axial') {
                // Check average value to determine Tension/Compression color
                const avgVal = (maxVal + minVal) / 2;
                if (avgVal < 0) {
                    // Compression -> Red
                    ctx.fillStyle = '#ef444433'; // Red 20%
                    ctx.strokeStyle = '#ef4444';
                    colorHex = '#ef4444';
                } else {
                    // Tension -> Blue
                    ctx.fillStyle = '#3b82f633';
                    ctx.strokeStyle = '#3b82f6';
                }
            } else {
                ctx.fillStyle = colorHex + '33';
                ctx.strokeStyle = colorHex;
            }

            ctx.fill();
            ctx.stroke();

            // Sign Label (+) or (-)
            if (areaSum > 0 && Math.abs(maxVal) > 1e-3) {
                 const cx = centroidX / areaSum;
                 const cy = centroidY / areaSum;
                 // Determine sign based on view mode rules
                 // For Moment: Sagging (+) is usually labeled (+).
                 // For Shear: Clockwise (+) is labeled (+).
                 // For Axial: Tension (+) is labeled (+).
                 // Just use the average value sign.
                 const avg = (maxVal + minVal) / 2;
                 const sign = avg >= 0 ? '+' : '-';
                 
                 // Draw Circle
                 ctx.beginPath();
                 ctx.arc(cx, cy, 8, 0, Math.PI*2);
                 ctx.fillStyle = '#ffffff';
                 ctx.fill();
                 ctx.strokeStyle = colorHex;
                 ctx.lineWidth = 1;
                 ctx.stroke();
                 
                 // Draw Text
                 ctx.fillStyle = colorHex;
                 ctx.font = 'bold 12px sans-serif';
                 ctx.textAlign = 'center';
                 ctx.textBaseline = 'middle';
                 ctx.fillText(sign, cx, cy + 1);
            }

            // Zero Line (Dashed)
            ctx.beginPath();
            ctx.strokeStyle = colorHex;
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 2;

            // --- LOD Labeling ---
            const unit = viewMode === 'moment' ? 'kN·m' : 'kN';
            const prefix = viewMode === 'moment' ? 'M' : (viewMode === 'shear' ? 'V' : 'N');
            
            // Always show Max/Min if significant
            if (Math.abs(maxVal) > 1e-3) {
                const off = getDrawOffset(maxVal, viewMode);
                const mx = p1.x + dx * maxPos.x + nx * off;
                const my = p1.y + dy * maxPos.x + ny * off;
                addLabel(`${maxVal.toFixed(2)}`, mx, my, colorHex, 8);
            }
            if (Math.abs(minVal) > 1e-3 && Math.abs(minVal - maxVal) > 1e-3) {
                const off = getDrawOffset(minVal, viewMode);
                const mx = p1.x + dx * minPos.x + nx * off;
                const my = p1.y + dy * minPos.x + ny * off;
                addLabel(`${minVal.toFixed(2)}`, mx, my, colorHex, 8);
            }

            // Hover Detail
            if (hovered && hovered.type === 'element' && hovered.id === el.id) {
                const t = hovered.t;
                const idx = points.findIndex(p => p.x >= t);
                const pt1 = points[Math.max(0, idx - 1)];
                const pt2 = points[idx] || points[points.length-1];
                
                const range = pt2.x - pt1.x;
                const localT = range === 0 ? 0 : (t - pt1.x) / range;
                
                const getVal = (p) => viewMode === 'moment' ? p.moment : (viewMode === 'shear' ? p.shear : p.axial);
                const v1 = getVal(pt1);
                const v2 = getVal(pt2);
                const val = v1 + (v2 - v1) * localT;

                const off = getDrawOffset(val, viewMode);
                const hx = p1.x + dx * t + nx * off;
                const hy = p1.y + dy * t + ny * off;

                // Draw Highlight Dot
                ctx.fillStyle = '#fff';
                ctx.strokeStyle = colorHex;
                ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI*2); ctx.fill(); ctx.stroke();

                // High Priority Label
                addLabel(`${prefix}=${val.toFixed(3)}`, hx, hy - 15, '#000', 10);
            }
        });
    }

    // --- Layer 4: Loads ---
    const loadAlpha = isResultView ? 0.2 : 1.0;
    
    // Element Loads
    renderElements.forEach(el => {
        const loads = renderElementLoads.filter(l => l.elementId === el.id);
        const n1 = renderNodes.find(n => n.id === el.n1), n2 = renderNodes.find(n => n.id === el.n2);
        if (!n1 || !n2) return;
        const p1 = toCanvas(n1.x, n1.y), p2 = toCanvas(n2.x, n2.y);
        const dx = p2.x-p1.x, dy = p2.y-p1.y, angle = Math.atan2(dy, dx), nx = -Math.sin(angle), ny = Math.cos(angle);
        
        loads.forEach(l => {
            ctx.globalAlpha = loadAlpha;
            ctx.strokeStyle = '#f97316'; ctx.fillStyle = '#f97316'; ctx.lineWidth = 1.5;

            // Reuse the Draw Arrow Helper (Defined inside loop for closure access to ctx)
             const drawArrow = (tail, tip) => {
                  ctx.beginPath(); ctx.moveTo(tail.x, tail.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
                  const headLen = 6;
                  const ang = Math.atan2(tip.y - tail.y, tip.x - tail.x);
                  ctx.beginPath();
                  ctx.moveTo(tip.x, tip.y);
                  ctx.lineTo(tip.x - headLen * Math.cos(ang - Math.PI/6), tip.y - headLen * Math.sin(ang - Math.PI/6));
                  ctx.lineTo(tip.x - headLen * Math.cos(ang + Math.PI/6), tip.y - headLen * Math.sin(ang + Math.PI/6));
                  ctx.fill();
             };

            if (l.type === 'uniform' || l.type === 'trapezoidal') {
                const h1 = (l.val1||0) * distLoadScalePxPerKNm;
                const h2 = (l.type==='uniform' ? (l.val1||0) : (l.val2||0)) * distLoadScalePxPerKNm;
                // Tail positions
                const t1x = p1.x - nx * h1, t1y = p1.y - ny * h1;
                const t2x = p2.x - nx * h2, t2y = p2.y - ny * h2;
                ctx.beginPath(); ctx.moveTo(t1x, t1y); ctx.lineTo(t2x, t2y); ctx.stroke();
                
                const steps = 8;
                for(let i=0; i<=steps; i++) {
                    const t = i/steps;
                    const tx = p1.x + dx*t, ty = p1.y + dy*t; // Point on beam
                    const h = h1 + (h2-h1)*t;
                    const tailX = tx - nx * h, tailY = ty - ny * h;
                    drawArrow({x: tailX, y: tailY}, {x: tx, y: ty});
                }
                if (!isResultView) addLabel(`q=${l.val1}`, t1x, t1y - 10, '#f97316', 4);

            } else if (l.type === 'point_force') {
                const L_el = Math.hypot(n2.x-n1.x, n2.y-n1.y);
                const px = p1.x + dx*(l.pos / L_el), py = p1.y + dy*(l.pos / L_el);
                const val = l.val1 || 0;
                
                let gfx = 0, gfy = 0;
                if (l.dir === 'axial') {
                    const ux = (n2.x-n1.x)/L_el, uy = (n2.y-n1.y)/L_el;
                    gfx = ux * val; gfy = uy * val;
                } else {
                    const ux = (n2.x-n1.x)/L_el, uy = (n2.y-n1.y)/L_el;
                    gfx = -uy * val; gfy = ux * val;
                }
                
                const vecLen = Math.min(40, Math.abs(val) * loadScalePxPerKN);
                const vMag = Math.hypot(gfx, gfy);
                const vx = (gfx/vMag) * vecLen;
                const vy = (-gfy/vMag) * vecLen; // Canvas Y flip
                
                const tail = { x: px - vx, y: py - vy };
                drawArrow(tail, {x: px, y: py});
                if (!isResultView) addLabel(`P=${val}`, tail.x, tail.y - 10, '#f97316', 5);
            }
        });
        ctx.globalAlpha = 1.0;
    });

    // Nodal Loads
    renderNodalLoads.forEach(l => {
        const n = renderNodes.find(node => node.id === l.nodeId);
        if (!n) return;
        const p = toCanvas(n.x, n.y);
        
        ctx.strokeStyle = '#dc2626'; ctx.fillStyle = '#dc2626'; ctx.lineWidth = 2;

        if (Math.abs(l.fx) > 0.1) {
             const len = Math.min(40, Math.abs(l.fx) * loadScalePxPerKN);
             const sign = Math.sign(l.fx);
             const tail = { x: p.x - sign*len, y: p.y };
             ctx.beginPath(); ctx.moveTo(tail.x, tail.y); ctx.lineTo(p.x, p.y); ctx.stroke();
             const headX = p.x - sign*6;
             ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(headX, p.y-3); ctx.lineTo(headX, p.y+3); ctx.fill();
             if (!isResultView) addLabel(`Fx=${l.fx}`, tail.x, tail.y - 10, '#dc2626', 5);
        }
        if (Math.abs(l.fy) > 0.1) {
             const len = Math.min(40, Math.abs(l.fy) * loadScalePxPerKN);
             const sign = Math.sign(l.fy);
             const tail = { x: p.x, y: p.y + sign*len }; // Canvas Coord
             ctx.beginPath(); ctx.moveTo(tail.x, tail.y); ctx.lineTo(p.x, p.y); ctx.stroke();
             const headY = p.y + sign*6; // Tip is p.y
             // Simple vertical arrow head
             ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x-3, headY); ctx.lineTo(p.x+3, headY); ctx.fill();
             if (!isResultView) addLabel(`Fy=${l.fy}`, tail.x + 5, tail.y, '#dc2626', 5);
        }
    });

    // --- Layer 5: Supports & Nodes ---
    renderNodes.forEach(n => {
      const p = toCanvas(n.x, n.y);
      const sup = supports.find(s => s.nodeId === n.id);
      if(sup) {
        const ang = (sup.angle || 0) * Math.PI / 180;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(-ang); // Visual rotation

        ctx.fillStyle = '#475569'; ctx.strokeStyle = '#475569'; ctx.lineWidth = 2;
        if(sup.type === 'fixed') {
          ctx.fillRect(-12, -2, 24, 6);
          for(let i=0; i<6; i++) { ctx.beginPath(); ctx.moveTo(-12+i*4, 4); ctx.lineTo(-16+i*4, 10); ctx.stroke(); }
        } else if(sup.type === 'pinned') {
          ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-10, 15); ctx.lineTo(10, 15); ctx.closePath(); ctx.fill();
        } else if(sup.type === 'rollerY') {
          ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-10, 15); ctx.lineTo(10, 15); ctx.closePath(); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(-12, 18); ctx.lineTo(12, 18); ctx.stroke();
        } else if(sup.type === 'rollerX') {
          ctx.save(); ctx.rotate(-Math.PI/2);
          ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-10, 15); ctx.lineTo(10, 15); ctx.closePath(); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(-12, 18); ctx.lineTo(12, 18); ctx.stroke();
          ctx.restore();
        }
        ctx.restore();
      }
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fillStyle = '#3b82f6'; ctx.fill(); ctx.stroke();
    });

    // Node Hover Logic
    if (hovered && hovered.type === 'node') {
        // Highlight circle
        ctx.beginPath(); 
        ctx.arc(hovered.x, hovered.y, 6, 0, 7); 
        ctx.strokeStyle = '#fff'; 
        ctx.lineWidth = 2; 
        ctx.stroke();
        
        // Add label
        addLabel(`Node ${hovered.id}`, hovered.x, hovered.y - 15, '#000', 10);
    }

    // --- Layer 6: Draw Labels (Sorted) ---
    labelList.sort((a, b) => b.priority - a.priority);
    labelList.forEach(l => drawSmartLabel(l));

  }, [nodes, elements, supports, elementLoads, nodalLoads, results, viewMode, transform, hovered]);

  return (
    <div className="flex flex-col h-screen bg-[#f8fafc] text-slate-800 font-sans overflow-hidden text-sm select-none">
      <header className="h-14 bg-white border-b px-6 flex items-center justify-between shadow-sm z-30">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-1.5 rounded-lg text-white shadow-lg"><Activity size={20} /></div>
          <h1 className="text-lg font-bold tracking-tight">SM Solver <span className="text-indigo-600 font-black">Ultra</span></h1>
        </div>
        <div className="flex gap-2 items-center">
          <div className="flex bg-slate-100 p-1 rounded-xl mr-4 shadow-inner">
            {['model', 'moment', 'shear', 'axial'].map(m => (
              <button key={m} onClick={() => setViewMode(m)} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all ${viewMode === m ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}>{m}</button>
            ))}
          </div>
          <button onClick={solve} className="bg-indigo-600 text-white px-6 py-2 rounded-xl text-xs font-bold shadow-lg hover:bg-indigo-700 transition-all flex items-center gap-2 active:scale-95"><Play size={14} fill="currentColor"/> 求解结构</button>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden">
        <aside className="w-[336px] min-w-[336px] max-w-[336px] flex-none bg-white border-r flex flex-col shadow-xl z-20 overflow-hidden">
          <div className="flex p-1.5 bg-slate-50 border-b gap-1">
            {[
              {id:'sections', icon: Layers, label:'截面库'},
              {id:'nodes', icon: Hash, label:'节点'},
              {id:'elements', icon: Layout, label:'单元'},
              {id:'supports', icon: Anchor, label:'支座'},
              {id:'loads', icon: Package, label:'荷载'}
            ].map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex-1 py-2 flex flex-col items-center gap-1 text-[9px] font-black uppercase rounded-lg transition-all ${activeTab === t.id ? 'bg-white shadow text-indigo-600 border border-slate-100' : 'text-slate-400'}`}>
                <t.icon size={14}/>{t.label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* 支座管理面板 */}
            {activeTab === 'supports' && (
              <div className="space-y-3">
                <button onClick={() => setSupports([...supports, { nodeId: nodes[0]?.id || 1, type: 'pinned' }])} className="w-full py-2.5 border-2 border-dashed border-slate-200 rounded-xl text-slate-400 font-bold hover:text-indigo-500 transition-all">+ 添加支座约束</button>
                {supports.map((sup, idx) => (
                  <div key={idx} className="bg-slate-50 p-3 rounded-2xl border border-slate-200 space-y-3 shadow-sm relative group">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-black text-slate-400 uppercase tracking-tighter">Support Constraint</span>
                      <button onClick={() => setSupports(supports.filter((_,i) => i !== idx))}><Trash2 size={12} className="text-slate-300 hover:text-red-500"/></button>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex flex-col gap-1">
                        <span className="text-[9px] text-slate-400">作用节点</span>
                        <select value={sup.nodeId} onChange={e => setSupports(supports.map((s,i) => i===idx ? {...s, nodeId:+e.target.value} : s))} className="p-1.5 rounded-lg border bg-white text-xs font-bold shadow-sm">
                          {nodes.map(n => <option key={n.id} value={n.id}>N{n.id} ({n.x}, {n.y})</option>)}
                        </select>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[9px] text-slate-400">支座类型</span>
                        <select value={sup.type} onChange={e => setSupports(supports.map((s,i) => i===idx ? {...s, type:e.target.value} : s))} className="p-1.5 rounded-lg border bg-white text-xs font-bold shadow-sm text-indigo-600">
                          <option value="fixed">固定端</option>
                          <option value="pinned">铰支座</option>
                          <option value="rollerY">滚动支座 (Y)</option>
                          <option value="rollerX">滚动支座 (X)</option>
                        </select>
                      </div>
                      <div className="col-span-2 flex items-center gap-2 bg-white p-1.5 rounded-lg border">
                          <span className="text-[9px] text-slate-400">旋转角度 (°)</span>
                          <input type="number" value={sup.angle || 0} onChange={e => setSupports(supports.map((s,i) => i===idx ? {...s, angle:+e.target.value} : s))} className="flex-1 text-right font-bold text-xs outline-none text-indigo-600"/>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === 'sections' && (
              <div className="space-y-3">
                <button onClick={() => setSections([...sections, {id: Date.now(), name: "新材料截面", E: 206000000, A: 0.01, I: 0.0002}])} className="w-full py-2.5 border-2 border-dashed border-slate-200 rounded-xl text-slate-400 font-bold hover:text-indigo-500">+ 新增截面定义</button>
                {sections.map(s => (
                  <div key={s.id} className="bg-slate-50 p-3 rounded-2xl border space-y-2 shadow-sm">
                    <div className="flex items-center gap-2"><input value={s.name} onChange={e => setSections(sections.map(x => x.id===s.id?{...x, name:e.target.value}:x))} className="flex-1 font-black bg-transparent outline-none text-indigo-700" /><button onClick={() => setSections(sections.filter(x => x.id !== s.id))}><Trash2 size={12}/></button></div>
                    <div className="grid grid-cols-1 gap-1 text-[10px] font-bold">
                      <div className="flex justify-between bg-white p-1 rounded border"><span>E (kN/m²)</span><input type="number" value={s.E} onChange={e => setSections(sections.map(x => x.id===s.id?{...x, E:+e.target.value}:x))} className="w-24 text-right outline-none" /></div>
                      <div className="flex justify-between bg-white p-1 rounded border"><span>A (m²)</span><input type="number" value={s.A} onChange={e => setSections(sections.map(x => x.id===s.id?{...x, A:+e.target.value}:x))} className="w-24 text-right outline-none" /></div>
                      <div className="flex justify-between bg-white p-1 rounded border"><span>I (m⁴)</span><input type="number" value={s.I} onChange={e => setSections(sections.map(x => x.id===s.id?{...x, I:+e.target.value}:x))} className="w-24 text-right outline-none" /></div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === 'elements' && (
              <div className="space-y-3">
                <button onClick={() => setElements([...elements, {id: Date.now(), n1: 1, n2: 2, sectionId: sections[0].id}])} className="w-full py-2.5 border-2 border-dashed rounded-xl text-slate-400 font-bold">+ 添加单元</button>
                {elements.map(el => (
                   <div key={el.id} className="bg-slate-50 p-3 rounded-2xl border space-y-2 shadow-sm">
                     <div className="flex justify-between text-[10px] font-black text-slate-300"><span>EL #{el.id % 1000}</span><button onClick={()=>setElements(elements.filter(x=>x.id!==el.id))}><Trash2 size={12}/></button></div>
                     <div className="grid grid-cols-2 gap-2">
                        <select value={el.n1} onChange={e=>setElements(elements.map(x=>x.id===el.id?{...x, n1:+e.target.value}:x))} className="p-1 border rounded bg-white">{nodes.map(n => <option key={n.id} value={n.id}>N{n.id}</option>)}</select>
                        <select value={el.n2} onChange={e=>setElements(elements.map(x=>x.id===el.id?{...x, n2:+e.target.value}:x))} className="p-1 border rounded bg-white">{nodes.map(n => <option key={n.id} value={n.id}>N{n.id}</option>)}</select>
                     </div>
                     <select value={el.sectionId} onChange={e=>setElements(elements.map(x=>x.id===el.id?{...x, sectionId:+e.target.value}:x))} className="w-full p-1 border rounded bg-indigo-50 font-bold text-indigo-600 shadow-inner">
                        {sections.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                     </select>
                     <div className="flex gap-3 items-center text-[9px] text-slate-400 font-bold bg-white p-1.5 rounded border">
                        <span className="text-slate-300">RELEASES</span>
                        <label className="flex items-center gap-1 cursor-pointer hover:text-indigo-600"><input type="checkbox" checked={el.releases?.[0]} onChange={e => {
                            const newRel = [...(el.releases || [0,0])]; newRel[0] = e.target.checked ? 1 : 0;
                            setElements(elements.map(x=>x.id===el.id?{...x, releases:newRel}:x));
                        }} className="accent-indigo-600"/> Start</label>
                        <label className="flex items-center gap-1 cursor-pointer hover:text-indigo-600"><input type="checkbox" checked={el.releases?.[1]} onChange={e => {
                            const newRel = [...(el.releases || [0,0])]; newRel[1] = e.target.checked ? 1 : 0;
                            setElements(elements.map(x=>x.id===el.id?{...x, releases:newRel}:x));
                        }} className="accent-indigo-600"/> End</label>
                     </div>
                   </div>
                ))}
              </div>
            )}

            {activeTab === 'loads' && (
              <div className="space-y-6">
                {/* 节点荷载部分 */}
                <div className="space-y-3">
                  <h4 className="font-black text-slate-300 uppercase tracking-widest text-[9px]">节点荷载 (Nodal Loads)</h4>
                  {nodalLoads.map((l) => (
                    <div key={l.id} className="bg-slate-50 p-3 rounded-2xl border space-y-2 shadow-sm">
                       <div className="flex justify-between items-center">
                          <select value={l.nodeId} onChange={e => setNodalLoads(nodalLoads.map(x=>x.id===l.id?{...x, nodeId:+e.target.value}:x))} className="font-black text-indigo-600 bg-transparent text-xs outline-none">
                            {nodes.map(n => <option key={n.id} value={n.id}>Node {n.id}</option>)}
                          </select>
                          <button onClick={() => setNodalLoads(nodalLoads.filter(x=>x.id!==l.id))} className="text-slate-300 hover:text-red-500"><Trash2 size={12}/></button>
                       </div>
                       <div className="grid grid-cols-3 gap-1">
                          <div className="flex items-center bg-white border rounded px-1"><span className="text-[9px] text-slate-400 mr-1">Fx (kN)</span><input type="number" value={l.fx} onChange={e => setNodalLoads(nodalLoads.map(x=>x.id===l.id?{...x, fx:+e.target.value}:x))} className="w-full text-xs outline-none" /></div>
                          <div className="flex items-center bg-white border rounded px-1"><span className="text-[9px] text-slate-400 mr-1">Fy (kN)</span><input type="number" value={l.fy} onChange={e => setNodalLoads(nodalLoads.map(x=>x.id===l.id?{...x, fy:+e.target.value}:x))} className="w-full text-xs outline-none" /></div>
                          <div className="flex items-center bg-white border rounded px-1"><span className="text-[9px] text-slate-400 mr-1">M (kN·m)</span><input type="number" value={l.m} onChange={e => setNodalLoads(nodalLoads.map(x=>x.id===l.id?{...x, m:+e.target.value}:x))} className="w-full text-xs outline-none" /></div>
                       </div>
                    </div>
                  ))}
                  <button onClick={() => {
                      const defaultNodeId = nodes.length > 0 ? nodes[0].id : 1;
                      setNodalLoads([...nodalLoads, {id: Date.now(), nodeId: defaultNodeId, fx:0, fy:-10, m:0}]);
                  }} className="w-full py-2 bg-indigo-50 text-indigo-600 rounded-xl font-black text-xs">+ 添加节点荷载</button>
                </div>

                {/* 单元荷载部分 */}
                <div className="space-y-3">
                  <h4 className="font-black text-slate-300 uppercase tracking-widest text-[9px]">单元荷载 (Element Loads)</h4>
                  {elementLoads.map((l) => (
                    <div key={l.id} className="bg-slate-50 p-3 rounded-2xl border space-y-2 shadow-sm">
                      <div className="flex justify-between items-center gap-2">
                        <select value={l.elementId} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, elementId:+e.target.value}:x))} className="font-black text-indigo-600 bg-transparent text-xs outline-none w-20">
                          {elements.map(el => <option key={el.id} value={el.id}>EL {el.id}</option>)}
                        </select>
                        <select value={l.type} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, type:e.target.value}:x))} className="flex-1 text-[10px] bg-white border border-slate-200 px-1 py-0.5 rounded font-bold">
                          <option value="uniform">均布 q</option>
                          <option value="trapezoidal">梯形 q1-q2</option>
                          <option value="point_force">集中力 P</option>
                          <option value="point_moment">集中弯矩 M</option>
                        </select>
                        <button onClick={() => setElementLoads(elementLoads.filter(x=>x.id!==l.id))} className="text-slate-300 hover:text-red-500"><Trash2 size={12}/></button>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        {/* 值输入 */}
                        <div className="col-span-2 flex gap-2">
                           <div className="flex-1 flex items-center bg-white border rounded px-1"><span className="text-[9px] text-slate-400 mr-1">{l.type==='point_moment' ? 'M (kN·m)' : l.type==='uniform' ? 'q (kN/m)' : l.type==='point_force' ? 'P (kN)' : l.type==='trapezoidal' ? 'q1 (kN/m)' : 'Val1'}</span><input type="number" value={l.val1} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, val1:+e.target.value}:x))} className="w-full text-xs outline-none" /></div>
                           {l.type === 'trapezoidal' && (
                             <div className="flex-1 flex items-center bg-white border rounded px-1"><span className="text-[9px] text-slate-400 mr-1">q2 (kN/m)</span><input type="number" value={l.val2} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, val2:+e.target.value}:x))} className="w-full text-xs outline-none" /></div>
                           )}
                        </div>

                        {/* 位置与方向 */}
                        {l.type !== 'uniform' && (
                          <div className="col-span-2 flex gap-2">
                             <div className="flex-1 flex items-center bg-indigo-50 border rounded px-1"><span className="text-[9px] text-indigo-300 mr-1">Pos (m)</span><input type="number" step="0.1" value={l.pos} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, pos:+e.target.value}:x))} className="w-full bg-transparent text-xs outline-none" /></div>
                             {l.type === 'point_force' && (
                               <select value={l.dir || 'vertical'} onChange={e => setElementLoads(elementLoads.map(x=>x.id===l.id?{...x, dir:e.target.value}:x))} className="flex-1 text-[10px] bg-white border rounded px-1">
                                 <option value="vertical">Vertical</option>
                                 <option value="axial">Axial</option>
                               </select>
                             )}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  <button onClick={() => {
                      const defaultElId = elements.length > 0 ? elements[0].id : 1;
                      setElementLoads([...elementLoads, {id:Date.now(), elementId:defaultElId, type:'uniform', val1:10}]);
                  }} className="w-full py-2 bg-indigo-50 text-indigo-600 rounded-xl font-black text-xs">+ 添加单元荷载</button>
                </div>
              </div>
            )}

            {activeTab === 'nodes' && (
              <div className="space-y-2">
                <button onClick={() => setNodes([...nodes, {id: nodes.length+1, x:0, y:0}])} className="w-full py-2 border-2 border-dashed rounded-xl text-slate-300 font-bold hover:text-indigo-500 transition-all">+ 添加节点</button>
                {nodes.map(n => (
                  <div key={n.id} className="flex gap-2 items-center bg-slate-50 p-2 rounded-xl border">
                    <span className="w-6 text-slate-300 font-black">#{n.id}</span>
                    <div className="flex items-center bg-white border rounded px-1 w-full">
                      <span className="text-[9px] text-slate-400 mr-1">X (m)</span>
                      <input type="number" value={n.x} onChange={e => setNodes(nodes.map(x=>x.id===n.id?{...x, x:+e.target.value}:x))} className="w-full text-xs outline-none" />
                    </div>
                    <div className="flex items-center bg-white border rounded px-1 w-full">
                      <span className="text-[9px] text-slate-400 mr-1">Y (m)</span>
                      <input type="number" value={n.y} onChange={e => setNodes(nodes.map(x=>x.id===n.id?{...x, y:+e.target.value}:x))} className="w-full text-xs outline-none" />
                    </div>
                    <button onClick={() => setNodes(nodes.filter(x=>x.id!==n.id))} className="text-slate-300 hover:text-red-500"><Trash2 size={14}/></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        <div className="flex-1 relative bg-[#f1f5f9] overflow-hidden">
          <div className="absolute top-6 right-6 flex flex-col gap-3 z-10">
            <button onClick={autoFit} className="bg-white p-3 rounded-2xl shadow-xl text-indigo-600 hover:bg-indigo-50 border border-indigo-50 flex items-center gap-2 font-black text-[10px] uppercase tracking-tighter transition-all active:scale-95"><Maximize2 size={18}/> Fit View</button>
            <div className="bg-white p-1 rounded-2xl shadow-xl flex flex-col border border-white">
              <button onClick={() => setTransform(prev => ({...prev, k: prev.k*1.2}))} className="p-3 text-slate-400 hover:text-indigo-600 transition-colors"><ZoomIn size={20}/></button>
              <button onClick={() => setTransform(prev => ({...prev, k: prev.k/1.2}))} className="p-3 text-slate-400 hover:text-indigo-600 transition-colors"><ZoomOut size={20}/></button>
            </div>
            <button onClick={exportData} className="bg-white p-3 rounded-2xl shadow-xl text-slate-400 hover:text-indigo-600 border border-white"><FileJson size={20}/></button>
          </div>

          <canvas 
            ref={canvasRef} 
            width={2000} 
            height={1600} 
            className={`w-full h-full cursor-${isPanning ? 'grabbing' : 'crosshair'}`}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onContextMenu={e => e.preventDefault()}
          />
          
          <div className="absolute bottom-6 left-6 flex gap-4 z-10">
             <div className="bg-white/80 backdrop-blur-md px-4 py-2 rounded-2xl shadow-2xl border border-white text-[10px] font-black text-slate-400 flex items-center gap-4">
                <span className="flex items-center gap-1.5"><Move size={12}/> PAN: RIGHT CLICK</span>
                <span className="flex items-center gap-1.5"><Maximize2 size={12}/> ZOOM: SCROLL</span>
             </div>
          </div>

          {results && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-white/95 backdrop-blur-md p-6 rounded-3xl shadow-2xl border border-white flex gap-10 items-center animate-in slide-in-from-bottom duration-500 w-[calc(100%-120px)] max-w-4xl">
               <div className="flex items-center gap-4">
                 <div className="bg-emerald-500 p-3 rounded-2xl text-white shadow-lg"><Activity size={24}/></div>
                 <div className="min-w-[100px]"><p className="text-[9px] font-black text-slate-300 uppercase leading-none mb-1">Status</p><p className="text-lg font-black text-slate-700 leading-none tracking-tighter">ANALYSIS READY</p></div>
               </div>
               <div className="h-10 w-px bg-slate-200" />
               <div className="flex-1 grid grid-cols-3 gap-6 font-mono font-bold text-[11px]">
                  <div><p className="text-slate-400 uppercase text-[9px] mb-1">MAX DISP</p><p className="text-lg text-indigo-600 tracking-tighter">{Math.max(...results.displacements.map(Math.abs)).toFixed(6)} m</p></div>
                  <div><p className="text-slate-400 uppercase text-[9px] mb-1">DOF</p><p className="text-lg text-slate-700 leading-none">{nodes.length * 3}</p></div>
                  <div><p className="text-slate-400 uppercase text-[9px] mb-1">UNITS</p><p className="text-lg text-slate-700 leading-none">kN / m</p></div>
               </div>
               <button onClick={exportData} className="bg-slate-900 text-white px-8 py-3.5 rounded-2xl font-black text-xs shadow-xl flex items-center gap-2 hover:bg-slate-800 transition-all active:scale-95"><Download size={18}/> DATA JSON</button>
            </div>
          )}

          {showDataCenter && (
            <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-12">
              <div className="bg-white w-full max-w-2xl rounded-3xl shadow-2xl flex flex-col overflow-hidden">
                <div className="p-5 border-b flex justify-between bg-slate-50 font-bold items-center text-slate-700 font-mono"><span className="flex items-center gap-2"><FileJson className="text-indigo-600" /> JSON DATA CENTER</span><button onClick={()=>setShowDataCenter(false)} className="text-slate-400 hover:text-slate-600">CLOSE</button></div>
                <textarea className="flex-1 p-6 font-mono text-xs text-indigo-800 outline-none resize-none h-80 bg-slate-50/50" value={importText} onChange={e => setImportText(e.target.value)} spellCheck="false" />
                <div className="p-4 border-t flex gap-3 justify-end bg-white"><button onClick={importData} className="bg-indigo-600 text-white px-8 py-2.5 rounded-2xl font-black text-xs shadow-lg active:scale-95">LOAD MODEL</button></div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer className="h-8 bg-white border-t px-6 flex items-center justify-between text-[10px] font-black text-slate-400 uppercase tracking-widest">
        <div className="flex gap-8">
          <span>NODES: {nodes.length}</span>
          <span>ELEMENTS: {elements.length}</span>
          <span>SECTIONS: {sections.length}</span>
          <span>SUPPORTS: {supports.length}</span>
        </div>
        <div className="flex items-center gap-4 text-indigo-400">
           <div className="flex items-center gap-1.5 font-black"><Hammer size={12}/> SYSTEM READY | {transform.k.toFixed(1)}x</div>
           <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"/> 
        </div>
      </footer>
    </div>
  );
};

export default App;