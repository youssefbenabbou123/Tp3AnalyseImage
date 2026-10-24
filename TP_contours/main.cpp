#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "laplacianZC.h"

namespace fs = std::filesystem;



struct ContourMetrics {
  int contours_detectes = 0;
      int contours_reference = 0;
   int contours_corrects = 0;
 int faux_positifs = 0;
     int faux_negatifs = 0;
  double P = 0.0;    
      double TFP = 0.0;
   double TFN = 0.0;
};


enum class DetectorType {
    Sobel = 0,               
  Laplace_Threshold = 1,   
      Laplace_ZeroCrossing = 2,
 Canny = 3                
};


struct AppState {
  int method_idx = 0;       
    int image_idx = 0;        
      int thresh1 = 100;        // meilleur seuil unique (Sobel) / seuil bas (Canny)
 int thresh2 = 300;        // seuil haut Canny (ratio 3:1 recommande OpenCV)
    int kernel_idx = 1;       
   int blur_idx = 0;         
      int use_neighborhood = 1; 

    std::vector<std::string> image_names;
  std::string images_dir;
      std::string gt_dir;

 cv::Mat current_img;
   cv::Mat current_gt;
      cv::Mat detected_binary;
 ContourMetrics current_metrics;
};

static AppState g_state;
static bool g_ui_ready = false;
static const std::string WINDOW_NAME = "TP3 - Detection et Evaluation Quantitative de Contours (BENABBOU Youssef)";

// trackbars
static const std::string TB_METHOD  = "Methode";
static const std::string TB_IMAGE   = "Image";
static const std::string TB_THRESH1 = "Seuil 1";
static const std::string TB_THRESH2 = "Seuil 2";
static const std::string TB_KERNEL  = "Noyau";
static const std::string TB_BLUR    = "Flou";
static const std::string TB_NEIGH   = "Voisin";

static cv::Rect g_chip_rects[4];
static cv::Rect g_btn_prev;
static cv::Rect g_btn_next;
static cv::Rect g_btn_voisin;

namespace Theme {
    const cv::Scalar BG_DARK        = cv::Scalar(18, 14, 12);
  const cv::Scalar CARD_BG        = cv::Scalar(32, 24, 20);
      const cv::Scalar CARD_BORDER    = cv::Scalar(58, 44, 38);
 const cv::Scalar HEADER_BG      = cv::Scalar(42, 32, 28);
    const cv::Scalar TEXT_MUTED     = cv::Scalar(160, 150, 140);
  const cv::Scalar TEXT_WHITE     = cv::Scalar(245, 245, 245);
      const cv::Scalar ACCENT_CYAN    = cv::Scalar(235, 180, 45);
 const cv::Scalar ACCENT_PURPLE  = cv::Scalar(215, 120, 160);
    const cv::Scalar ACCENT_GREEN   = cv::Scalar(75, 215, 80);
      const cv::Scalar ACCENT_RED     = cv::Scalar(55, 65, 240);
  const cv::Scalar ACCENT_BLUE    = cv::Scalar(245, 160, 25);
 const cv::Scalar ACCENT_GOLD    = cv::Scalar(60, 195, 255);
}

int getKernelSize(int idx) {
    switch (idx) {
      case 0: return 1;
        case 1: return 3;
     case 2: return 5;
          case 3: return 7;
        default: return 3;
    }
}

int getBlurSize(int idx) {
  switch (idx) {
        case 1: return 3;
     case 2: return 5;
          default: return 0;
  }
}

std::string getMethodName(int idx) {
    switch (idx) {
      case 0: return "Sobel (Norme L1 - derivatives.cpp)";
        case 1: return "Laplace (Seuillage Amplitude)";
     case 2: return "Laplace (Passage par Zero / ZC)";
          case 3: return "Canny (Optimal Hysteresis - contours.cpp)";
        default: return "Inconnu";
    }
}

std::string getMethodShortName(int idx) {
  switch (idx) {
        case 0: return "SOBEL L1";
     case 1: return "LAPLACE SEUIL";
          case 2: return "LAPLACE ZC";
        case 3: return "CANNY OPTIMAL";
     default: return "DET";
  }
}

cv::Scalar getMethodBadgeColor(int idx) {
    switch (idx) {
      case 0: return Theme::ACCENT_CYAN;
        case 1: return Theme::ACCENT_PURPLE;
     case 2: return cv::Scalar(220, 80, 200);
          case 3: return Theme::ACCENT_GOLD;
        default: return Theme::ACCENT_CYAN;
    }
}

// fonction pour appliquer les filtres
cv::Mat applyDetector(const cv::Mat& src, int method_idx, int th1, int th2, int ksize, int blur_sz) {
    cv::Mat gray;
    src.copyTo(gray);  // copie profonde obligatoire : evite la modification de src en batch
    if (blur_sz > 0) {
        cv::GaussianBlur(gray, gray, cv::Size(blur_sz, blur_sz), 0);
    }

cv::Mat output = cv::Mat::zeros(src.size(), CV_8U);

    if (method_idx == 0) {
      // sobel
      cv::Mat sobelX, sobelY;
        cv::Sobel(gray, sobelX, CV_16S, 1, 0, ksize);
    cv::Sobel(gray, sobelY, CV_16S, 0, 1, ksize);

      cv::Mat sobelMag = cv::abs(sobelX) + cv::abs(sobelY);
        double smin, smax;
    cv::minMaxLoc(sobelMag, &smin, &smax);

        cv::Mat sobelNorm;
      if (smax > 0) {
          sobelMag.convertTo(sobelNorm, CV_8U, 255.0 / smax, 0);
      } else {
        sobelNorm = cv::Mat::zeros(gray.size(), CV_8U);
      }
    cv::threshold(sobelNorm, output, th1, 255, cv::THRESH_BINARY);
    }
  else if (method_idx == 1) {
    cv::Mat laplaceMat;
      cv::Laplacian(gray, laplaceMat, CV_16S, ksize);
    cv::Mat lapMag = cv::abs(laplaceMat);

        double lmin, lmax;
      cv::minMaxLoc(lapMag, &lmin, &lmax);

    cv::Mat lapNorm;
      if (lmax > 0) {
        lapMag.convertTo(lapNorm, CV_8U, 255.0 / lmax, 0);
      } else {
          lapNorm = cv::Mat::zeros(gray.size(), CV_8U);
      }
    cv::threshold(lapNorm, output, th1, 255, cv::THRESH_BINARY);
  }
    else if (method_idx == 2) {
      // laplace zc
        LaplacianZC lapZC; //32f 
     lapZC.setAperture(ksize); 
        cv::Mat flap = lapZC.computeLaplacian(gray);
      output = lapZC.getZeroCrossings(flap);
    }
  else if (method_idx == 3) {
      // canny 
    cv::Canny(gray, output, th1, th2, ksize);
  }

    return output;
}

// calcul metriques
ContourMetrics computeMetrics(const cv::Mat& detected, const cv::Mat& gt_binary, bool use_neighborhood) {
  ContourMetrics m;
    cv::Mat det_bin = (detected > 0);
 cv::Mat gt_bin  = (gt_binary > 0);

  m.contours_detectes  = cv::countNonZero(det_bin);
      m.contours_reference = cv::countNonZero(gt_bin);

    cv::Mat correct_mask;
  if (use_neighborhood) {
      cv::Mat gt_dilated;
        cv::dilate(gt_bin, gt_dilated, cv::Mat());
    cv::bitwise_and(det_bin, gt_dilated, correct_mask);
  } else {
      cv::bitwise_and(det_bin, gt_bin, correct_mask);
  }

 m.contours_corrects = cv::countNonZero(correct_mask);
    m.faux_positifs     = std::max(0, m.contours_detectes - m.contours_corrects);
  m.faux_negatifs     = std::max(0, m.contours_reference - m.contours_corrects);

      int denom = m.contours_corrects + m.faux_positifs + m.faux_negatifs;
  if (denom > 0) {
      m.P   = static_cast<double>(m.contours_corrects) / denom;
    m.TFP = static_cast<double>(m.faux_positifs) / denom;
        m.TFN = static_cast<double>(m.faux_negatifs) / denom;
  } else {
    m.P = 0.0;
        m.TFP = 0.0;
    m.TFN = 0.0;
  }

    return m;
}

// overlay diagnostique
cv::Mat createOverlay(const cv::Mat& src, const cv::Mat& det_binary, const cv::Mat& gt_binary, bool use_neighborhood) {
  cv::Mat color;
    cv::cvtColor(src, color, cv::COLOR_GRAY2BGR);

 cv::Mat overlay;
  cv::addWeighted(color, 0.40, cv::Mat(src.size(), CV_8UC3, cv::Scalar(24, 20, 18)), 0.60, 0.0, overlay);

    cv::Mat det_bin = (det_binary > 0);
  cv::Mat gt_bin  = (gt_binary > 0);

      cv::Mat gt_dilated;
  if (use_neighborhood) {
      cv::dilate(gt_bin, gt_dilated, cv::Mat());
  } else {
    gt_dilated = gt_bin.clone();
  }

 cv::Mat corr_mask;
    cv::bitwise_and(det_bin, gt_dilated, corr_mask);

  cv::Mat fp_mask;
    cv::bitwise_and(det_bin, ~corr_mask, fp_mask);

      cv::Mat det_dilated;
  if (use_neighborhood) {
    cv::dilate(det_bin, det_dilated, cv::Mat());
  } else {
      det_dilated = det_bin.clone();
  }

    cv::Mat fn_mask;
  cv::bitwise_and(gt_bin, ~det_dilated, fn_mask);

 overlay.setTo(Theme::ACCENT_BLUE, fn_mask);    
  overlay.setTo(Theme::ACCENT_RED, fp_mask);     
      overlay.setTo(Theme::ACCENT_GREEN, corr_mask); 

  return overlay;
}

// dessin cartes
void renderPanelCard(cv::Mat& canvas, int x, int y, int card_w, int card_h,
                     const cv::Mat& img_content, const std::string& badge_txt,
                     const cv::Scalar& badge_col, const std::string& title_txt) {
  cv::Rect card_rect(x, y, card_w, card_h);
    cv::rectangle(canvas, card_rect, Theme::CARD_BG, cv::FILLED);
 cv::rectangle(canvas, card_rect, Theme::CARD_BORDER, 1);

    int header_h = 24;
  cv::Rect header_rect(x, y, card_w, header_h);
      cv::rectangle(canvas, header_rect, Theme::HEADER_BG, cv::FILLED);
  cv::line(canvas, cv::Point(x, y + header_h), cv::Point(x + card_w, y + header_h), Theme::CARD_BORDER, 1);

 int badge_w = 80;
    int badge_h = 16;
  cv::Rect badge_rect(x + 6, y + 4, badge_w, badge_h);
      cv::rectangle(canvas, badge_rect, badge_col, cv::FILLED);
  cv::putText(canvas, badge_txt, cv::Point(x + 10, y + 16), cv::FONT_HERSHEY_DUPLEX, 0.35, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);

 cv::putText(canvas, title_txt, cv::Point(x + badge_w + 12, y + 17), cv::FONT_HERSHEY_SIMPLEX, 0.40, Theme::TEXT_WHITE, 1, cv::LINE_AA);

    int img_sz = card_h - header_h - 10;
  int img_x = x + (card_w - img_sz) / 2;
      int img_y = y + header_h + 5;

    cv::Mat resized;
  cv::resize(img_content, resized, cv::Size(img_sz, img_sz));

 cv::Mat bgr_content;
  if (resized.channels() == 1) {
      cv::cvtColor(resized, bgr_content, cv::COLOR_GRAY2BGR);
  } else {
    bgr_content = resized;
  }

    cv::Rect img_roi(img_x, img_y, img_sz, img_sz);
  bgr_content.copyTo(canvas(img_roi));
      cv::rectangle(canvas, img_roi, Theme::CARD_BORDER, 1);
}

// kpi
void renderKPICard(cv::Mat& canvas, int x, int y, int w, int h,
                   const std::string& label, const std::string& value,
                   const cv::Scalar& val_col, const std::string& subtext) {
    cv::Rect r(x, y, w, h);
  cv::rectangle(canvas, r, Theme::CARD_BG, cv::FILLED);
      cv::rectangle(canvas, r, Theme::CARD_BORDER, 1);

 cv::rectangle(canvas, cv::Rect(x, y, 4, h), val_col, cv::FILLED);
    cv::putText(canvas, label, cv::Point(x + 10, y + 15), cv::FONT_HERSHEY_DUPLEX, 0.35, Theme::TEXT_MUTED, 1, cv::LINE_AA);
  cv::putText(canvas, value, cv::Point(x + 10, y + 38), cv::FONT_HERSHEY_DUPLEX, 0.65, val_col, 2, cv::LINE_AA);
      cv::putText(canvas, subtext, cv::Point(x + 10, y + 51), cv::FONT_HERSHEY_SIMPLEX, 0.32, Theme::TEXT_MUTED, 1, cv::LINE_AA);
}

// tableau de bord
cv::Mat createDashboard(const AppState& state) {
  const int TOTAL_W = 900;
      const int CARD_W  = 435;
 const int CARD_H  = 240;
    const int MARGIN  = 10;

  int header_h = 56;
      int hud_h    = 108;
 int total_h  = MARGIN + header_h + 6 + CARD_H + 6 + CARD_H + 6 + hud_h + MARGIN;

    cv::Mat dash(total_h, TOTAL_W, CV_8UC3, Theme::BG_DARK);

  cv::Rect top_rect(MARGIN, 4, TOTAL_W - 2 * MARGIN, header_h);
      cv::rectangle(dash, top_rect, Theme::HEADER_BG, cv::FILLED);
  cv::rectangle(dash, top_rect, Theme::CARD_BORDER, 1);

 cv::putText(dash, "TP3 - EVALUATION QUANTITATIVE DE CONTOURS", cv::Point(MARGIN + 12, 20),
             cv::FONT_HERSHEY_DUPLEX, 0.44, Theme::TEXT_WHITE, 1, cv::LINE_AA);

    g_btn_prev = cv::Rect(TOTAL_W - MARGIN - 330, 7, 72, 18);
  g_btn_next = cv::Rect(TOTAL_W - MARGIN - 250, 7, 72, 18);

      cv::rectangle(dash, g_btn_prev, cv::Scalar(50, 40, 36), cv::FILLED);
  cv::rectangle(dash, g_btn_prev, Theme::CARD_BORDER, 1);
 cv::putText(dash, "< PREV", cv::Point(g_btn_prev.x + 8, g_btn_prev.y + 13),
             cv::FONT_HERSHEY_SIMPLEX, 0.38, Theme::TEXT_WHITE, 1, cv::LINE_AA);

    cv::rectangle(dash, g_btn_next, cv::Scalar(50, 40, 36), cv::FILLED);
  cv::rectangle(dash, g_btn_next, Theme::CARD_BORDER, 1);
      cv::putText(dash, "NEXT >", cv::Point(g_btn_next.x + 10, g_btn_next.y + 13),
                  cv::FONT_HERSHEY_SIMPLEX, 0.38, Theme::TEXT_WHITE, 1, cv::LINE_AA);

  std::string author_str = "BENABBOU Youssef";
      cv::putText(dash, author_str, cv::Point(TOTAL_W - MARGIN - 165, 20),
                  cv::FONT_HERSHEY_DUPLEX, 0.40, Theme::ACCENT_CYAN, 1, cv::LINE_AA);

    std::vector<std::string> chip_names = {"[0] Sobel L1", "[1] Laplace Seuil", "[2] Laplace ZC", "[3] Canny Opt"};
  int chip_x = MARGIN + 10;
      int chip_y = 28;
 int chip_w = (TOTAL_W - 2 * MARGIN - 20 - 3 * 10) / 4;
    int chip_h = 24;

  for (int i = 0; i < 4; ++i) {
      g_chip_rects[i] = cv::Rect(chip_x + i * (chip_w + 10), chip_y, chip_w, chip_h);
    bool is_active = (i == state.method_idx);
        cv::Scalar fill_col = is_active ? getMethodBadgeColor(i) : cv::Scalar(28, 22, 20);
    cv::Scalar txt_col = is_active ? cv::Scalar(10, 10, 10) : Theme::TEXT_MUTED;

        cv::rectangle(dash, g_chip_rects[i], fill_col, cv::FILLED);
    cv::rectangle(dash, g_chip_rects[i], is_active ? cv::Scalar(255, 255, 255) : Theme::CARD_BORDER, is_active ? 2 : 1);
        cv::putText(dash, chip_names[i], cv::Point(g_chip_rects[i].x + 18, g_chip_rects[i].y + 16),
                    cv::FONT_HERSHEY_DUPLEX, 0.40, txt_col, is_active ? 2 : 1, cv::LINE_AA);
  }

    int y_row1 = MARGIN + header_h + 6;
  int y_row2 = y_row1 + CARD_H + 6;
      int x_col1 = MARGIN;
 int x_col2 = MARGIN + CARD_W + 10;

  std::string img_name = state.image_names.empty() ? "N/A" : state.image_names[state.image_idx];

  // source
    renderPanelCard(dash, x_col1, y_row1, CARD_W, CARD_H,
                    state.current_img, "1. SOURCE", Theme::ACCENT_CYAN,
                    img_name + " (512x512 PGM)");

  // detecte
    cv::Mat det_inv = 255 - state.detected_binary;
  std::string det_title = getMethodShortName(state.method_idx) + " | " + std::to_string(state.current_metrics.contours_detectes) + " px";
      renderPanelCard(dash, x_col2, y_row1, CARD_W, CARD_H,
                      det_inv, "2. DETECTE", getMethodBadgeColor(state.method_idx),
                      det_title);

  // ref
  std::string gt_title = "Reference RUG | " + std::to_string(state.current_metrics.contours_reference) + " px";
      renderPanelCard(dash, x_col1, y_row2, CARD_W, CARD_H,
                      state.current_gt, "3. REF (GT)", Theme::ACCENT_GREEN,
                      gt_title);

  // erreur
    cv::Mat overlay = createOverlay(state.current_img, state.detected_binary, state.current_gt == 0, state.use_neighborhood != 0);
  renderPanelCard(dash, x_col2, y_row2, CARD_W, CARD_H,
                  overlay, "4. ERREUR", Theme::ACCENT_RED,
                  "V=Correct | R=FP | B=FN");

      int y_hud = y_row2 + CARD_H + 6;
  int hud_w = TOTAL_W - 2 * MARGIN;
    cv::Rect hud_rect(MARGIN, y_hud, hud_w, hud_h);
 cv::rectangle(dash, hud_rect, cv::Scalar(24, 18, 16), cv::FILLED);
    cv::rectangle(dash, hud_rect, Theme::CARD_BORDER, 1);

  int b1_w = 150, b_h = 20;
      cv::Rect b1_r(MARGIN + 10, y_hud + 6, b1_w, b_h);
  cv::rectangle(dash, b1_r, getMethodBadgeColor(state.method_idx), cv::FILLED);
      cv::putText(dash, getMethodShortName(state.method_idx), cv::Point(b1_r.x + 8, b1_r.y + 14),
                  cv::FONT_HERSHEY_DUPLEX, 0.38, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);

 cv::Scalar tol_col = state.use_neighborhood ? Theme::ACCENT_GREEN : cv::Scalar(40, 140, 240);
    std::string tol_str = state.use_neighborhood ? "VOISINAGE 3x3 (ACTIF)" : "TOLERANCE STRICTE 1x1";
  int b2_w = 175;
      g_btn_voisin = cv::Rect(b1_r.x + b1_w + 8, y_hud + 6, b2_w, b_h);
  cv::rectangle(dash, g_btn_voisin, tol_col, cv::FILLED);
      cv::rectangle(dash, g_btn_voisin, cv::Scalar(255, 255, 255), 1);
  cv::putText(dash, tol_str, cv::Point(g_btn_voisin.x + 8, g_btn_voisin.y + 14),
              cv::FONT_HERSHEY_DUPLEX, 0.36, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);

    char param_buf[128];
  snprintf(param_buf, sizeof(param_buf), "Img [%d/40]: %s | S1=%d S2=%d K=%d G=%d",
           state.image_idx + 1, img_name.c_str(), state.thresh1, state.thresh2,
           getKernelSize(state.kernel_idx), getBlurSize(state.blur_idx));
      cv::putText(dash, param_buf, cv::Point(g_btn_voisin.x + b2_w + 14, y_hud + 20),
                  cv::FONT_HERSHEY_SIMPLEX, 0.38, Theme::TEXT_MUTED, 1, cv::LINE_AA);

  int kpi_y = y_hud + 32;
      int kpi_h = 56;
 int kpi_gap = 8;
    int kpi_w = (hud_w - 20 - 3 * kpi_gap) / 4;

  char buf[64];
      snprintf(buf, sizeof(buf), "%.3f", state.current_metrics.P);
  cv::Scalar p_col = (state.current_metrics.P >= 0.35) ? Theme::ACCENT_GREEN :
                     (state.current_metrics.P >= 0.20) ? Theme::ACCENT_GOLD : cv::Scalar(100, 120, 240);
      renderKPICard(dash, MARGIN + 10, kpi_y, kpi_w, kpi_h,
                    "PERFORMANCE (P)", buf, p_col, "Score Qualite Global");

  snprintf(buf, sizeof(buf), "%.3f", state.current_metrics.TFP);
      renderKPICard(dash, MARGIN + 10 + (kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                    "TAUX FAUX POS. (TFP)", buf, Theme::ACCENT_RED, "Bruit / Sur-detection");

    snprintf(buf, sizeof(buf), "%.3f", state.current_metrics.TFN);
  renderKPICard(dash, MARGIN + 10 + 2 * (kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                "TAUX FAUX NEG. (TFN)", buf, Theme::ACCENT_BLUE, "Contours Manques");

      int kpi4_x = MARGIN + 10 + 3 * (kpi_w + kpi_gap);
  cv::Rect r4(kpi4_x, kpi_y, kpi_w, kpi_h);
      cv::rectangle(dash, r4, Theme::CARD_BG, cv::FILLED);
  cv::rectangle(dash, r4, Theme::CARD_BORDER, 1);
      cv::rectangle(dash, cv::Rect(kpi4_x, kpi_y, 4, kpi_h), Theme::TEXT_MUTED, cv::FILLED);

 cv::putText(dash, "PIXELS COMPTES", cv::Point(kpi4_x + 8, kpi_y + 13), cv::FONT_HERSHEY_DUPLEX, 0.30, Theme::TEXT_MUTED, 1, cv::LINE_AA);
    snprintf(buf, sizeof(buf), "Ref : %d px", state.current_metrics.contours_reference);
  cv::putText(dash, buf, cv::Point(kpi4_x + 8, kpi_y + 26), cv::FONT_HERSHEY_SIMPLEX, 0.33, Theme::TEXT_WHITE, 1, cv::LINE_AA);
      snprintf(buf, sizeof(buf), "Corr: %d | FP: %d", state.current_metrics.contours_corrects, state.current_metrics.faux_positifs);
  cv::putText(dash, buf, cv::Point(kpi4_x + 8, kpi_y + 39), cv::FONT_HERSHEY_SIMPLEX, 0.31, Theme::ACCENT_GREEN, 1, cv::LINE_AA);
      snprintf(buf, sizeof(buf), "FN  : %d px", state.current_metrics.faux_negatifs);
  cv::putText(dash, buf, cv::Point(kpi4_x + 8, kpi_y + 51), cv::FONT_HERSHEY_SIMPLEX, 0.31, Theme::ACCENT_BLUE, 1, cv::LINE_AA);

    int y_help = y_hud + hud_h - 7;
  std::string help_str = "CLIC SOURIS: [0..4] Methode | [< / >] Image | [Voisinage]  ---  CLAVIER: [0..4], [N/P] Image, [V], [B] Batch, [Q] Quitter";
      cv::putText(dash, help_str, cv::Point(MARGIN + 12, y_help), cv::FONT_HERSHEY_SIMPLEX, 0.33, Theme::TEXT_MUTED, 1, cv::LINE_AA);

  return dash;
}

void updateView();
void setMethodDefaults(int method_idx);

// gestion souris
void onMouse(int event, int x, int y, int flags, void*) {
    if (!g_ui_ready) return;

  if (event == cv::EVENT_LBUTTONDOWN) {
      for (int i = 0; i < 4; ++i) {
        if (g_chip_rects[i].contains(cv::Point(x, y))) {
            g_state.method_idx = i;
          cv::setTrackbarPos(TB_METHOD, WINDOW_NAME, i);
              setMethodDefaults(i);
          updateView();
              return;
        }
      }

    if (g_btn_prev.contains(cv::Point(x, y))) {
        g_state.image_idx = (g_state.image_idx - 1 + static_cast<int>(g_state.image_names.size())) % g_state.image_names.size();
      cv::setTrackbarPos(TB_IMAGE, WINDOW_NAME, g_state.image_idx);
          updateView();
      return;
    }

      if (g_btn_next.contains(cv::Point(x, y))) {
        g_state.image_idx = (g_state.image_idx + 1) % g_state.image_names.size();
          cv::setTrackbarPos(TB_IMAGE, WINDOW_NAME, g_state.image_idx);
        updateView();
          return;
      }

    if (g_btn_voisin.contains(cv::Point(x, y))) {
        g_state.use_neighborhood = 1 - g_state.use_neighborhood;
      cv::setTrackbarPos(TB_NEIGH, WINDOW_NAME, g_state.use_neighborhood);
          updateView();
      return;
    }
  }
    else if (event == cv::EVENT_MOUSEWHEEL) {
    int delta = cv::getMouseWheelDelta(flags);
        if (delta > 0) {
        g_state.image_idx = (g_state.image_idx + 1) % g_state.image_names.size();
        } else if (delta < 0) {
          g_state.image_idx = (g_state.image_idx - 1 + static_cast<int>(g_state.image_names.size())) % g_state.image_names.size();
        }
      cv::setTrackbarPos(TB_IMAGE, WINDOW_NAME, g_state.image_idx);
        updateView();
  }
}

// affichage
void updateView() {
  if (g_state.image_names.empty()) return;

    std::string img_name = g_state.image_names[g_state.image_idx];
  std::string img_path = g_state.images_dir + "/" + img_name + ".pgm";
      std::string gt_path = g_state.gt_dir + "/" + img_name + "_gt_binary.pgm";

  g_state.current_img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
      g_state.current_gt  = cv::imread(gt_path, cv::IMREAD_GRAYSCALE);

  if (g_state.current_img.empty() || g_state.current_gt.empty()) {
        std::cerr << "[-] Erreur chargement image : " << img_name << std::endl;
    return;
  }

    int ksize = getKernelSize(g_state.kernel_idx);
  int blur_sz = getBlurSize(g_state.blur_idx);

      g_state.detected_binary = applyDetector(
        g_state.current_img,
        g_state.method_idx,
          g_state.thresh1,
        g_state.thresh2,
          ksize,
        blur_sz
      );

  cv::Mat gt_binary = (g_state.current_gt == 0);

      g_state.current_metrics = computeMetrics(
        g_state.detected_binary,
        gt_binary,
          g_state.use_neighborhood != 0
      );

  cv::Mat db = createDashboard(g_state);
      cv::imshow(WINDOW_NAME, db);
}

// defaults optimaux par methode
void setMethodDefaults(int method_idx) {
  if (method_idx == 0) {
      // Sobel : seuil 125 sur gradient normalise [0-255]
    g_state.thresh1 = 125;
        g_state.thresh2 = 0;
    g_state.kernel_idx = 1;
        g_state.blur_idx = 1;
  } else if (method_idx == 1) {
      // Laplace Seuil : seuil 125 (meme que Sobel pour comparaison equitable)
    g_state.thresh1 = 125;
        g_state.thresh2 = 0;
    g_state.kernel_idx = 1;
        g_state.blur_idx = 1;
  } else if (method_idx == 2) {
      // Laplace ZC : pas de seuil (zero crossing)
    g_state.thresh1 = 0;
        g_state.thresh2 = 0;
    g_state.kernel_idx = 1;
        g_state.blur_idx = 1;
  } else if (method_idx == 3) {
      // Canny : S1=125 (meme seuil bas que Sobel), S2=350 (seuil haut optimal)
    g_state.thresh1 = 125;
        g_state.thresh2 = 350;
    g_state.kernel_idx = 1;
        g_state.blur_idx = 1;
  }

  if (g_ui_ready) {
      cv::setTrackbarPos(TB_THRESH1, WINDOW_NAME, g_state.thresh1);
    cv::setTrackbarPos(TB_THRESH2, WINDOW_NAME, g_state.thresh2);
        cv::setTrackbarPos(TB_KERNEL,  WINDOW_NAME, g_state.kernel_idx);
    cv::setTrackbarPos(TB_BLUR,    WINDOW_NAME, g_state.blur_idx);
  }
}

void onMethodTrackbar(int pos, void*) {
  if (!g_ui_ready) return;
      g_state.method_idx = pos;
  setMethodDefaults(pos);
      updateView();
}

void onTrackbar(int, void*) {
    if (!g_ui_ready) return;
  updateView();
}

// batch
void runBatchProcessing(const AppState& state) {
  std::cout << "\n========================================================" << std::endl;
      std::cout << "[+] DEMARRAGE DU TRAITEMENT PAR LOT (40 IMAGES BENCHMARK)" << std::endl;
  std::cout << "========================================================" << std::endl;

    std::string out_csv = "resultats_lot_cpp.csv";
  std::ofstream file(out_csv);
      if (!file.is_open()) {
    std::cerr << "[-] Impossible d'ecrire le fichier : " << out_csv << std::endl;
        return;
      }

  file << "image,methode,ref,det,corr,fp,fn,P,TFP,TFN,voisinage\n";

      std::vector<std::string> methods = {"Sobel", "Laplace", "LaplaceZC", "Canny"};

  struct MethodStats {
      double p_sum = 0.0;
    double tfp_sum = 0.0;
        double tfn_sum = 0.0;
    int count = 0;
  };

      std::vector<MethodStats> stats(4);

  std::cout << std::setw(14) << "Image"
            << std::setw(12) << "Methode"
            << std::setw(8)  << "Ref"
            << std::setw(8)  << "Det"
            << std::setw(8)  << "Corr"
            << std::setw(8)  << "FP"
            << std::setw(8)  << "FN"
            << std::setw(10) << "P(3x3)"
            << std::setw(10) << "TFP"
            << std::setw(10) << "TFN" << std::endl;
  std::cout << std::string(96, '-') << std::endl;

      for (const auto& name : state.image_names) {
    std::string img_path = state.images_dir + "/" + name + ".pgm";
        std::string gt_path = state.gt_dir + "/" + name + "_gt_binary.pgm";

    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(gt_path, cv::IMREAD_GRAYSCALE);
    if (img.empty() || gt.empty()) continue;

        cv::Mat gt_bin = (gt == 0);

    for (int m = 0; m < 4; ++m) {
        int th1 = (m == 3) ? 125 : 125;  // Sobel=125, Laplace=125, LaplaceZC=0 (ignored), Canny=125
      int th2 = (m == 3) ? 350 : 0;
          int ksz = 3;
      int blur_sz = 3; // lissage gaussien 3x3 pour tous les detecteurs (comparaison equitable)

          cv::Mat det = applyDetector(img, m, th1, th2, ksz, blur_sz);
      ContourMetrics met = computeMetrics(det, gt_bin, true);

        file << name << "," << methods[m] << ","
             << met.contours_reference << ","
            << met.contours_detectes << ","
            << met.contours_corrects << ","
             << met.faux_positifs << ","
            << met.faux_negatifs << ","
            << met.P << ","
            << met.TFP << ","
             << met.TFN << ",3x3\n";

      stats[m].p_sum += met.P;
          stats[m].tfp_sum += met.TFP;
      stats[m].tfn_sum += met.TFN;
          stats[m].count++;

      if (name == "elephant" || name == "turtle" || name == "buffalo" || name == "bear" || name == "basket") {
          std::cout << std::setw(14) << name
                    << std::setw(12) << methods[m]
                   << std::setw(8)  << met.contours_reference
                   << std::setw(8)  << met.contours_detectes
                    << std::setw(8)  << met.contours_corrects
                   << std::setw(8)  << met.faux_positifs
                    << std::setw(8)  << met.faux_negatifs
                   << std::setw(10) << std::fixed << std::setprecision(3) << met.P
                    << std::setw(10) << met.TFP
                   << std::setw(10) << met.TFN << std::endl;
      }
    }
  }

    file.close();
  std::cout << std::string(96, '-') << std::endl;
      std::cout << "\n--- MOYENNES GLOBALES SUR LES 40 IMAGES DU BENCHMARK ---" << std::endl;
  for (int m = 0; m < 4; ++m) {
      double p_avg = stats[m].p_sum / stats[m].count;
    double tfp_avg = stats[m].tfp_sum / stats[m].count;
        double tfn_avg = stats[m].tfn_sum / stats[m].count;
    std::cout << "[" << std::setw(10) << methods[m] << "] P moyen = " << std::fixed << std::setprecision(4) << p_avg
              << " | TFP moyen = " << tfp_avg
              << " | TFN moyen = " << tfn_avg << std::endl;
  }
      std::cout << "[+] Fichier CSV sauvegarde : " << out_csv << std::endl;
}

// dataset
bool locateDataset(AppState& state) {
  std::vector<std::string> candidates = {
      "../contours/images",
    "../../contours/images",
        "contours/images",
    "c:/Users/Josef/Desktop/analyse d'image/contours/images"
  };

  for (const auto& c : candidates) {
      if (fs::exists(c) && fs::exists(c + "/gt")) {
        state.images_dir = fs::canonical(c).string();
            state.gt_dir = fs::canonical(c + "/gt").string();
        break;
      }
  }

  if (state.images_dir.empty()) {
        std::cerr << "[-] Impossible de localiser le dossier contours/images !" << std::endl;
    return false;
  }

      std::cout << "[+] Dossier images : " << state.images_dir << std::endl;
  std::cout << "[+] Dossier GT     : " << state.gt_dir << std::endl;

  for (const auto& entry : fs::directory_iterator(state.images_dir)) {
      if (entry.path().extension() == ".pgm") {
        std::string stem = entry.path().stem().string();
            std::string gt_file = state.gt_dir + "/" + stem + "_gt_binary.pgm";
        if (fs::exists(gt_file)) {
            state.image_names.push_back(stem);
        }
      }
  }

      std::sort(state.image_names.begin(), state.image_names.end());
  std::cout << "* Images valides avec GT chargees : " << state.image_names.size() << std::endl;
      return !state.image_names.empty();
}

int main(int argc, char** argv) {
  
      std::cout << "  TP3 - DETECTION ET EVALUATION QUANTITATIVE DE CONTOURS   \n";
  std::cout << "  Etudiant : Youssef BENABBOU                              \n";
      std::cout << "  OpenCV 4.x - Sobel, Laplace, Canny                      \n";
 

  if (!locateDataset(g_state)) {
        return -1;
  }

  if (argc > 1 && std::string(argv[1]) == "--batch") {
      runBatchProcessing(g_state);
    return 0;
  }

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  cv::createTrackbar(TB_METHOD,  WINDOW_NAME, &g_state.method_idx, 3, onMethodTrackbar);
      int max_img = static_cast<int>(g_state.image_names.size()) - 1;   
  cv::createTrackbar(TB_IMAGE,   WINDOW_NAME, &g_state.image_idx, max_img, onTrackbar);

      cv::createTrackbar(TB_THRESH1, WINDOW_NAME, &g_state.thresh1, 255, onTrackbar);
  cv::createTrackbar(TB_THRESH2, WINDOW_NAME, &g_state.thresh2, 500, onTrackbar);
      cv::createTrackbar(TB_KERNEL,  WINDOW_NAME, &g_state.kernel_idx, 3, onTrackbar);
  cv::createTrackbar(TB_BLUR,    WINDOW_NAME, &g_state.blur_idx, 2, onTrackbar);
      cv::createTrackbar(TB_NEIGH,   WINDOW_NAME, &g_state.use_neighborhood, 1, onTrackbar);

  cv::setMouseCallback(WINDOW_NAME, onMouse, nullptr);

      g_ui_ready = true;

  std::cout << "\n--- CONTROLES ---" << std::endl;
      std::cout << "  [0..3] : Changer Methode (Sobel, Laplace Seuil, ZC, Canny)" << std::endl;
  std::cout << "  [N / P] : Image Suivante / Precedente (ou molette souris)" << std::endl;
      std::cout << "  [V]     : Basculer Voisinage 3x3 / Stricte" << std::endl;
  std::cout << "  [B]     : Lancer Traitement par lot (batch)" << std::endl;
      std::cout << "  [Q/ESC] : Quitter\n" << std::endl;

  setMethodDefaults(g_state.method_idx); // applique les seuils par defaut au demarrage
  updateView();

  while (true) {
      int key = cv::waitKey(30);
    if (key == 27 || key == 'q' || key == 'Q') {
          break;
    } else if (key >= '0' && key <= '3') {
        int m = key - '0';
      g_state.method_idx = m;
          cv::setTrackbarPos(TB_METHOD, WINDOW_NAME, m);
      setMethodDefaults(m);
          updateView();
    } else if (key == 'n' || key == 'N' || key == 32) {
        g_state.image_idx = (g_state.image_idx + 1) % g_state.image_names.size();
      cv::setTrackbarPos(TB_IMAGE, WINDOW_NAME, g_state.image_idx);
          updateView();
    } else if (key == 'p' || key == 'P') {
        g_state.image_idx = (g_state.image_idx - 1 + static_cast<int>(g_state.image_names.size())) % g_state.image_names.size();
      cv::setTrackbarPos(TB_IMAGE, WINDOW_NAME, g_state.image_idx);
          updateView();
    } else if (key == 'm' || key == 'M') {
        g_state.method_idx = (g_state.method_idx + 1) % 4;
      cv::setTrackbarPos(TB_METHOD, WINDOW_NAME, g_state.method_idx);
          setMethodDefaults(g_state.method_idx);
      updateView();
    } else if (key == 'v' || key == 'V') {
        g_state.use_neighborhood = 1 - g_state.use_neighborhood;
      cv::setTrackbarPos(TB_NEIGH, WINDOW_NAME, g_state.use_neighborhood);
          updateView();
    } else if (key == 'b' || key == 'B') {
        runBatchProcessing(g_state);
    }
  }

  return 0;
}
