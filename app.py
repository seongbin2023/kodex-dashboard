/**
 * 글로벌 3대 지수 및 핵심 환율(달러, 엔화) 통합 분석 엔진입니다.
 * 실시간 함수가 과거 데이터를 덮어쓰는 문제를 해결한 '날짜 고정형' 버전입니다.
 */
function runGlobalAnalysis() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const assets = [
    ["KODEX200_분석", "KRX:069500", "KODEX 200"],
    ["SP500_분석", "INDEXSP:.INX", "S&P 500"],
    ["NASDAQ_분석", "INDEXNASDAQ:.IXIC", "NASDAQ"],
    ["USD_환율_분석", "CURRENCY:USDKRW", "달러 환율 (USD/KRW)"],
    ["JPY_환율_분석", "CURRENCY:JPYKRW", "엔화 환율 (100JPY/KRW)"]
  ];

  assets.forEach(asset => {
    const [sheetName, ticker, displayName] = asset;
    let sheet = ss.getSheetByName(sheetName);
    
    if (!sheet) {
      sheet = ss.insertSheet(sheetName);
      sheet.appendRow(["날짜", "현재가", "200일선", "이격률(%)", "상태 판정"]);
      sheet.getRange("A1:E1").setBackground("#fff2cc").setFontWeight("bold");
    }

    const today = new Date();
    const dateStr = Utilities.formatDate(today, "GMT+9", "yyyy-MM-dd");
    
    const lastRow = sheet.getLastRow();
    let targetRow = lastRow + 1;
    if (lastRow > 1) {
      const lastDateInSheet = sheet.getRange(lastRow, 1).getDisplayValue();
      if (lastDateInSheet === dateStr) {
        targetRow = lastRow; 
      }
    }

    // 수정된 부분: A열의 날짜(A${targetRow})를 기준으로 해당 일자의 종가를 가져옵니다.
    // IFERROR를 사용하여 당일 데이터가 덜 닫혔을 때는 실시간 현재가를 임시로 사용합니다.
    let priceFormula = `=IFERROR(INDEX(GOOGLEFINANCE("${ticker}", "price", A${targetRow}), 2, 2), GOOGLEFINANCE("${ticker}"))`;
    let maFormula = `=AVERAGE(INDEX(GOOGLEFINANCE("${ticker}", "price", A${targetRow}-300, A${targetRow}), 0, 2))`;
    
    if (ticker === "CURRENCY:JPYKRW") {
      priceFormula = `=IFERROR(INDEX(GOOGLEFINANCE("${ticker}", "price", A${targetRow}), 2, 2), GOOGLEFINANCE("${ticker}")) * 100`;
      maFormula = `=AVERAGE(INDEX(GOOGLEFINANCE("${ticker}", "price", A${targetRow}-300, A${targetRow}), 0, 2)) * 100`;
    }

    sheet.getRange(targetRow, 1).setValue(dateStr);
    sheet.getRange(targetRow, 2).setFormula(priceFormula);
    sheet.getRange(targetRow, 3).setFormula(maFormula);
    sheet.getRange(targetRow, 4).setFormula(`=(B${targetRow}-C${targetRow})/C${targetRow}*100`);
    
    if (ticker.includes("CURRENCY")) {
      sheet.getRange(targetRow, 5).setFormula(`=IF(B${targetRow}>C${targetRow}, "[강세/상승]", "⚠️[약세/하락]")`);
    } else {
      sheet.getRange(targetRow, 5).setFormula(`=IF(B${targetRow}>C${targetRow}, "[안전]", "⚠️[위험]")`);
    }

    updateGlobalChart(sheet, displayName);
  });
}

function updateGlobalChart(sheet, title) {
  const charts = sheet.getCharts();
  for (let i in charts) { sheet.removeChart(charts[i]); }

  const chart = sheet.newChart()
    .setChartType(Charts.ChartType.LINE)
    .addRange(sheet.getRange("A1:A" + sheet.getLastRow()))
    .addRange(sheet.getRange("D1:D" + sheet.getLastRow()))
    .setPosition(2, 7, 0, 0)
    .setOption('title', `${title} 이격률(Disparity) 추이 (%)`)
    .build();
    
  sheet.insertChart(chart);
}
