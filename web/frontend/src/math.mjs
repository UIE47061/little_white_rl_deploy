// Robot and rendering both use X-forward, Y-left, Z-up. No extra axis rotation.
export function imuQuaternion(wxyz) {
  const [w, x, y, z] = wxyz;
  const norm = Math.hypot(w, x, y, z);
  if (!Number.isFinite(norm) || norm < 1e-6) throw new Error('Invalid IMU quaternion');
  return [x / norm, y / norm, z / norm, w / norm];
}
export function deadzone(x, y, threshold = 0.1) {
  const radius = Math.hypot(x, y);
  if (radius <= threshold) return [0, 0];
  const scale = (Math.min(1, radius) - threshold) / ((1 - threshold) * radius);
  return [x * scale, y * scale];
}
export function jointValues(names, angles) {
  if (names.length !== angles.length) throw new Error('Joint count mismatch');
  return Object.fromEntries(names.map((name, i) => [name, angles[i]]));
}
