(use-modules (meep))
(use-modules (meep core) (meep geom))

;;; CONSTANTS ;;;
(define pi (* 2 (acos 0)))  ; pi
(define ix_SiN 1.9935)  ; index of Si3N4

;;; PHC DIMENSIONS ;;;
(define amp 120)  ; nominal amplitude (nm)
(define wid 280)  ; nominal width (nm)
(define thk 200)  ; nominal thickness (nm)
(define gap 238)  ; nominal gap (nm)
(define a 370)  ; lattice constant (nm)

;;; AUX SETTINGS ;;;
(define resolution 37)  ; number of blocks per lattice constant
(define dx (/ 1 resolution))  ; thickness of blocks; sub-resolution

(set! filename-prefix  ; filename prefix
  (string-append
    "w" (number->string wid)
    "A" (number->string amp)
    "g" (number->string gap)
  )
)

;;; CELL DIMENSIONS ;;;
(define sx 1)  ; lattice constant in lattice constants
(define sy 15)  ; y dimension in lattice constants
(define sz 15)  ; z dimension in lattice constants
(set! geometry-lattice (make lattice (size sx sy sz)))

;;; DEFINE UNIT CONVERSION FUNCTIONS ;;;
(define (simUnits x) (/ x a))

;;; CREATE GEOMETRY ;;;
(define (drawBlock x)
  (let (
    (dy (+ wid (* amp (cos (* 2 pi x)))))  ; block width
  )
    (let (
      (cy (/ (+ dy gap) 2))  ; block center
    )
      (list
        ;; Top block
        (make block 
          (center x (simUnits cy) 0)
          (size (* 1.1 dx) (simUnits dy) (simUnits thk))
          (material (make dielectric (index ix_SiN)))
        )
        ;; Bottom block
        (make block 
          (center x (* -1 (simUnits cy)) 0)
          (size (* 1.1 dx) (simUnits dy) (simUnits thk))
          (material (make dielectric (index ix_SiN)))
        )
      )  ;; End of list
    )  ;; End of inner let
  )  ;; End of outer let
)

(set! geometry '())  ; initialize empty geometry

(define (makeGeom x)
  (if (>= x 1)
      '()  ; exit loop
      (begin
        (set! geometry (append geometry (drawBlock x)))
        (makeGeom (+ x dx))
      )
  )
)

(makeGeom 0)  ; recursively fill geometry

;;; RUN
;; set k-points, num-bands
(set! k-points (list (vector3 0.5 0 0)))  ; only calculate band edge
(set! num-bands 2)

;;; RUN ;;;
(run-yodd-zeven)  ; TE mode
